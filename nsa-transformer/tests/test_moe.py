import torch
import pytest
from models.moe import MoELayer, ExpertMLP, Router

@pytest.fixture
def moe_config():
    return {
        "d_model": 256,
        "n_experts": 8,
        "expert_dim": 1024,
        "top_k": 2,
        "capacity_factor": 1.25,
        "dropout": 0.1,
    }

def test_expert_mlp():
    """Test individual expert MLP."""
    expert = ExpertMLP(d_model=256, d_hidden=1024)
    x = torch.randn(32, 128, 256)  # [batch, seq, d_model]
    
    out = expert(x)
    assert out.shape == x.shape
    
def test_router(moe_config):
    """Test expert routing mechanism."""
    router = Router(
        d_model=moe_config["d_model"],
        n_experts=moe_config["n_experts"],
        top_k=moe_config["top_k"],
    )
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, moe_config["d_model"])
    
    # Get routing probabilities and expert assignments
    route_probs, indices, combine_weights = router(x)
    
    # Check shapes
    assert route_probs.shape == (batch_size * seq_len, moe_config["n_experts"])
    assert indices.shape == (batch_size * seq_len, moe_config["top_k"])
    assert combine_weights.shape == (batch_size * seq_len, moe_config["top_k"])
    
    # Check routing properties
    assert torch.all(route_probs >= 0) and torch.all(route_probs <= 1)
    assert torch.all(indices >= 0) and torch.all(indices < moe_config["n_experts"])
    assert torch.allclose(combine_weights.sum(dim=-1), torch.ones_like(combine_weights.sum(dim=-1)))
    
def test_moe_layer(moe_config):
    """Test complete MoE layer."""
    moe = MoELayer(**moe_config)
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, moe_config["d_model"])
    
    # Forward pass
    out = moe(x)
    assert out.shape == x.shape
    
def test_load_balancing(moe_config):
    """Test expert load balancing."""
    moe = MoELayer(**moe_config)
    batch_size, seq_len = 8, 512  # Larger batch for better statistics
    x = torch.randn(batch_size, seq_len, moe_config["d_model"])
    
    # Get expert counts
    route_probs = moe.router(x)[0]
    expert_counts = route_probs.argmax(dim=-1).bincount(
        minlength=moe_config["n_experts"]
    )
    
    # Check load balancing
    total_tokens = batch_size * seq_len
    expected_count = total_tokens / moe_config["n_experts"]
    max_imbalance = 0.3  # Allow 30% imbalance
    
    assert torch.all(
        (expert_counts >= expected_count * (1 - max_imbalance)) &
        (expert_counts <= expected_count * (1 + max_imbalance))
    )
    
def test_capacity_limiting(moe_config):
    """Test expert capacity limiting."""
    moe = MoELayer(**moe_config)
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, moe_config["d_model"])
    
    # Calculate maximum tokens per expert
    total_tokens = batch_size * seq_len
    max_expert_capacity = int(
        total_tokens * moe_config["capacity_factor"] / moe_config["n_experts"]
    )
    
    # Forward pass with expert assignment tracking
    with torch.no_grad():
        moe(x)
        expert_counts = moe._expert_counts  # Assuming we track this in forward
        
    # Check capacity limits
    assert torch.all(expert_counts <= max_expert_capacity)
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_expert_offloading(moe_config):
    """Test expert CPU offloading."""
    moe = MoELayer(**moe_config).cuda()
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, moe_config["d_model"]).cuda()
    
    # Move experts to CPU
    for expert in moe.experts:
        expert.to("cpu")
        
    # Forward pass should automatically handle expert movement
    out = moe(x)
    assert out.device == x.device  # Output should be on GPU
    
    # Check experts are back on CPU
    for expert in moe.experts:
        assert next(expert.parameters()).device == torch.device("cpu") 