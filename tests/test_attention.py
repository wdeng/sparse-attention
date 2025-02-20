import torch
import pytest
from models.attention import NSAttention, CompressionMLP

@pytest.fixture
def attention_config():
    return {
        "d_model": 256,
        "n_heads": 8,
        "block_size": 16,
        "compression_stride": 8,
        "top_blocks": 4,
        "window_size": 64,
        "gqa_groups": 2,
        "dropout": 0.1,
    }

def test_compression_mlp():
    """Test block compression MLP."""
    mlp = CompressionMLP(d_model=256, block_size=16)
    x = torch.randn(2, 4, 16, 256)  # [batch, blocks, block_size, d_model]
    
    out = mlp(x)
    assert out.shape == (2, 4, 256)  # [batch, blocks, d_model]
    
def test_attention_shapes(attention_config):
    """Test output shapes of attention mechanism."""
    attn = NSAttention(**attention_config)
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, attention_config["d_model"])
    
    out = attn(x)
    assert out.shape == (batch_size, seq_len, attention_config["d_model"])
    
def test_attention_mask(attention_config):
    """Test attention masking."""
    attn = NSAttention(**attention_config)
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, attention_config["d_model"])
    
    # Create attention mask (mask out second half of sequence)
    mask = torch.ones(batch_size, seq_len)
    mask[:, seq_len//2:] = 0
    
    out_masked = attn(x, mask)
    out_unmasked = attn(x)
    
    # Check that masked and unmasked outputs differ
    assert not torch.allclose(out_masked, out_unmasked)
    
def test_gqa_groups(attention_config):
    """Test grouped query attention."""
    attn = NSAttention(**attention_config)
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, attention_config["d_model"])
    
    # Check that number of key/value heads is reduced by GQA groups
    q = attn._reshape_heads(attn.q_proj(x))
    k = attn._reshape_heads(attn.k_proj(x), is_key_value=True)
    v = attn._reshape_heads(attn.v_proj(x), is_key_value=True)
    
    assert q.shape[1] == attention_config["n_heads"]  # Full heads for queries
    assert k.shape[1] == attention_config["gqa_groups"]  # Reduced heads for keys
    assert v.shape[1] == attention_config["gqa_groups"]  # Reduced heads for values
    
def test_branch_gating(attention_config):
    """Test attention branch gating mechanism."""
    attn = NSAttention(**attention_config)
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, attention_config["d_model"])
    
    # Get gating weights
    gates = attn.gate_mlp(x)  # [batch, seq, 3]
    
    # Check gate properties
    assert gates.shape == (batch_size, seq_len, 3)
    assert torch.all(gates >= 0) and torch.all(gates <= 1)  # Sigmoid output
    assert torch.allclose(gates.sum(dim=-1), torch.ones_like(gates.sum(dim=-1)))
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_kernels(attention_config):
    """Test CUDA kernel implementation."""
    attn = NSAttention(**attention_config).cuda()
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, attention_config["d_model"]).cuda()
    
    # Run with both PyTorch and CUDA implementations
    try:
        import sparse_attention_cuda
        attn._use_cuda_kernels = True
        out_cuda = attn(x)
        
        attn._use_cuda_kernels = False
        out_pytorch = attn(x)
        
        # Check outputs are similar (allowing for small numerical differences)
        assert torch.allclose(out_cuda, out_pytorch, rtol=1e-3, atol=1e-3)
    except ImportError:
        pytest.skip("CUDA kernels not available") 