import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import wandb
from typing import Dict, Optional, List
import hydra
from omegaconf import DictConfig
import numpy as np

from models.transformer import NSATransformerBlock
from data.dataloader import PackedDataLoader
from .optim import create_optimizer_scheduler
from utils.logging import WandBLogger
from utils.memory import (
    setup_memory_optimizations,
    quantize_kv_cache,
    setup_expert_offloading,
)

class Trainer:
    """Distributed trainer with FSDP and memory optimizations."""
    
    def __init__(
        self,
        config: DictConfig,
        model: torch.nn.Module,
        train_dataloader: PackedDataLoader,
        local_rank: int,
        world_size: int,
    ):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Setup mixed precision
        self.mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        # Setup FSDP wrapping policy
        self.auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={NSATransformerBlock},
        )
        
        # Setup FSDP model
        self.model = FSDP(
            model,
            mixed_precision=self.mixed_precision,
            auto_wrap_policy=self.auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=True),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
        )
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_scheduler(
            self.model,
            config.training.optimizer,
            config.training.scheduler,
        )
        
        # Setup data loader
        self.train_dataloader = train_dataloader
        
        # Setup memory optimizations
        setup_memory_optimizations(
            model=self.model,
            config=config.memory,
            local_rank=local_rank,
        )
        
        # Setup expert offloading if using MoE
        if config.model.moe_layers:
            setup_expert_offloading(
                model=self.model,
                config=config.memory.expert_offloading,
            )
            
        # Setup logging
        self.logger = WandBLogger(config) if local_rank == 0 else None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def _forward_backward_step(
        self,
        batch: Dict[str, torch.Tensor],
        grad_accum_steps: int,
    ) -> float:
        """Perform forward and backward pass."""
        # Move batch to device
        batch = {k: v.to(self.local_rank) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs[1] / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * grad_accum_steps
        
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Scheduler step
        self.scheduler.step()
        
    def _save_checkpoint(self, loss: float):
        """Save training checkpoint."""
        if self.local_rank == 0:
            checkpoint = {
                'step': self.step,
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': loss,
            }
            
            # Save checkpoint
            checkpoint_dir = os.path.join(
                self.config.output_dir,
                'checkpoints',
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'checkpoint_{self.step}.pt'),
            )
            
            # Save best model
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, 'best_model.pt'),
                )
                
    def _log_metrics(self, loss: float, lr: float):
        """Log training metrics."""
        if self.local_rank == 0 and self.logger is not None:
            metrics = {
                'train/loss': loss,
                'train/lr': lr,
                'train/step': self.step,
                'train/epoch': self.epoch,
            }
            self.logger.log_metrics(metrics, step=self.step)
            
    def train(self):
        """Main training loop."""
        self.model.train()
        
        # Training loop
        while self.step < self.config.training.max_steps:
            running_loss = 0.0
            
            # Gradient accumulation loop
            for i in range(self.config.training.gradient_accumulation_steps):
                # Get batch
                batch = next(iter(self.train_dataloader))
                
                # Forward and backward pass
                loss = self._forward_backward_step(
                    batch,
                    self.config.training.gradient_accumulation_steps,
                )
                running_loss += loss
                
            # Optimizer step
            self._optimizer_step()
            
            # Update step counter
            self.step += 1
            
            # Calculate metrics
            avg_loss = running_loss / self.config.training.gradient_accumulation_steps
            lr = self.scheduler.get_last_lr()[0]
            
            # Log metrics
            self._log_metrics(avg_loss, lr)
            
            # Save checkpoint
            if self.step % self.config.logging.checkpointing.save_steps == 0:
                self._save_checkpoint(avg_loss)
                
            # Update epoch counter
            if self.step % self.config.training.steps_per_epoch == 0:
                self.epoch += 1
                
@hydra.main(config_path="../configs", config_name="experiment")
def main(config: DictConfig):
    # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    # Create model and data loader
    model = hydra.utils.instantiate(config.model)
    train_dataloader = hydra.utils.instantiate(config.data.train_dataloader)
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    # Start training
    trainer.train()
    
if __name__ == "__main__":
    main() 