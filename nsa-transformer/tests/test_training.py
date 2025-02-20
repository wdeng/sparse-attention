import torch
import pytest
from training.trainer import Trainer
from training.optim import CosineSchedulerWithWarmup
from omegaconf import OmegaConf
import os
import tempfile

@pytest.fixture
def config():
    return OmegaConf.create({
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "moe_layers": [1, 3],
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 1000,
            "steps_per_epoch": 100,
            "max_grad_norm": 1.0,
            "optimizer": {
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
            "scheduler": {
                "warmup_steps": 100,
                "max_steps": 1000,
                "min_lr": 1e-5,
            },
        },
        "memory": {
            "kv_cache": {
                "quantization": "fp8",
                "threshold": 512,
            },
            "activation_checkpointing": {
                "granularity": "selective",
                "checkpoint_every_n_layers": 2,
            },
            "expert_offloading": {
                "enabled": True,
                "pin_memory": True,
            },
        },
        "logging": {
            "wandb": {
                "project": "test_project",
                "entity": "test_entity",
                "log_interval": 10,
            },
            "checkpointing": {
                "save_steps": 100,
            },
        },
        "output_dir": tempfile.mkdtemp(),
    })

@pytest.fixture
def dummy_model():
    """Create dummy model for testing."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(256, 256)
            
        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.layer(input_ids)
            loss = torch.nn.functional.mse_loss(outputs, labels) if labels is not None else None
            return (outputs, loss)
            
    return DummyModel()

@pytest.fixture
def dummy_dataloader():
    """Create dummy dataloader for testing."""
    class DummyDataLoader:
        def __iter__(self):
            while True:
                batch = {
                    "input_ids": torch.randn(2, 128, 256),
                    "attention_mask": torch.ones(2, 128),
                    "labels": torch.randn(2, 128, 256),
                }
                yield batch
                
    return DummyDataLoader()

def test_trainer_initialization(config, dummy_model, dummy_dataloader):
    """Test trainer initialization."""
    trainer = Trainer(
        config=config,
        model=dummy_model,
        train_dataloader=dummy_dataloader,
        local_rank=0,
        world_size=1,
    )
    
    assert isinstance(trainer.model, torch.nn.Module)
    assert hasattr(trainer, "optimizer")
    assert hasattr(trainer, "scheduler")
    
def test_forward_backward_step(config, dummy_model, dummy_dataloader):
    """Test forward and backward pass."""
    trainer = Trainer(
        config=config,
        model=dummy_model,
        train_dataloader=dummy_dataloader,
        local_rank=0,
        world_size=1,
    )
    
    batch = next(iter(dummy_dataloader))
    loss = trainer._forward_backward_step(batch, grad_accum_steps=1)
    
    assert isinstance(loss, float)
    assert loss > 0
    
def test_optimizer_step(config, dummy_model, dummy_dataloader):
    """Test optimizer step with gradient clipping."""
    trainer = Trainer(
        config=config,
        model=dummy_model,
        train_dataloader=dummy_dataloader,
        local_rank=0,
        world_size=1,
    )
    
    # Get initial parameters
    initial_params = [p.clone() for p in trainer.model.parameters()]
    
    # Perform optimization step
    batch = next(iter(dummy_dataloader))
    trainer._forward_backward_step(batch, grad_accum_steps=1)
    trainer._optimizer_step()
    
    # Check parameters have been updated
    for p, p_init in zip(trainer.model.parameters(), initial_params):
        assert not torch.allclose(p, p_init)
        
def test_gradient_accumulation(config, dummy_model, dummy_dataloader):
    """Test gradient accumulation."""
    trainer = Trainer(
        config=config,
        model=dummy_model,
        train_dataloader=dummy_dataloader,
        local_rank=0,
        world_size=1,
    )
    
    # Get gradients with and without accumulation
    def get_grads(accum_steps):
        trainer.optimizer.zero_grad()
        batch = next(iter(dummy_dataloader))
        loss = trainer._forward_backward_step(batch, grad_accum_steps=accum_steps)
        return [p.grad.clone() for p in trainer.model.parameters() if p.grad is not None]
        
    grads_no_accum = get_grads(accum_steps=1)
    grads_accum = get_grads(accum_steps=4)
    
    # Check gradients are scaled properly
    for g_no_accum, g_accum in zip(grads_no_accum, grads_accum):
        assert torch.allclose(g_no_accum, g_accum * 4, rtol=1e-3)
        
def test_checkpointing(config, dummy_model, dummy_dataloader):
    """Test model checkpointing."""
    trainer = Trainer(
        config=config,
        model=dummy_model,
        train_dataloader=dummy_dataloader,
        local_rank=0,
        world_size=1,
    )
    
    # Train for a few steps
    for _ in range(5):
        batch = next(iter(dummy_dataloader))
        trainer._forward_backward_step(batch, grad_accum_steps=1)
        trainer._optimizer_step()
        
    # Save checkpoint
    trainer._save_checkpoint(loss=0.1)
    
    # Check checkpoint files exist
    checkpoint_path = os.path.join(config.output_dir, "checkpoints", "checkpoint_0.pt")
    assert os.path.exists(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision(config, dummy_model, dummy_dataloader):
    """Test mixed precision training."""
    trainer = Trainer(
        config=config,
        model=dummy_model.cuda(),
        train_dataloader=dummy_dataloader,
        local_rank=0,
        world_size=1,
    )
    
    # Check model parameters are in mixed precision
    for p in trainer.model.parameters():
        assert p.dtype == torch.bfloat16
        
    # Forward pass should work with mixed precision
    batch = next(iter(dummy_dataloader))
    loss = trainer._forward_backward_step(batch, grad_accum_steps=1)
    assert isinstance(loss, float) 