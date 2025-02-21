import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import sparse_attention_cuda
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    print("Warning: CUDA kernels not available, falling back to PyTorch implementation")
    CUDA_KERNELS_AVAILABLE = False

class CompressionMLP(nn.Module):
    """Block compression MLP that maps sequence blocks to compressed representations."""
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.layer1 = nn.Linear(block_size * d_model, 4 * d_model)
        self.layer2 = nn.Linear(4 * d_model, d_model)
        self.act = nn.GELU()
        
        # Add positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, 1, block_size, d_model))
        
        # Add dtype tracking for mixed precision
        self.register_buffer('_dummy', torch.empty(0), persistent=False)
        
    @property
    def dtype(self) -> torch.dtype:
        return self._dummy.dtype
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, blocks, block_size, d_model]
        batch, blocks, block_size, d_model = x.shape
        
        # Handle mixed precision
        comp_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        x = x.to(comp_dtype)
        
        # Add positional embeddings
        x = x + self.pos_emb.to(comp_dtype)
        
        with torch.cuda.amp.autocast(enabled=True):
            x = x.reshape(batch, blocks, -1)  # [batch, blocks, block_size * d_model]
            x = self.act(self.layer1(x))
            x = self.layer2(x)  # [batch, blocks, d_model]
            
        return x

class NSAttention(nn.Module):
    """Neural Sparse Attention implementing three parallel branches with gating."""
    
    def __init__(
        self,
        d_model: int = 2560,
        n_heads: int = 32,
        block_size: int = 32,
        compression_stride: int = 16,
        top_blocks: int = 16,
        window_size: int = 512,
        gqa_groups: int = 4,
        dropout: float = 0.1,
        use_fp8_kv_cache: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        assert n_heads % gqa_groups == 0, "Number of heads must be divisible by GQA groups"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.compression_stride = compression_stride
        self.top_blocks = top_blocks
        self.window_size = window_size
        self.gqa_groups = gqa_groups
        self.use_fp8_kv_cache = use_fp8_kv_cache
        self.gradient_checkpointing = gradient_checkpointing
        
        # Projection matrices
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Branch-specific components
        self.compression_mlp = CompressionMLP(d_model, block_size)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # FP8 scaling factors for KV cache quantization
        if use_fp8_kv_cache:
            self.register_buffer('k_scale', torch.ones(1))
            self.register_buffer('v_scale', torch.ones(1))
            
        # Add dtype tracking for mixed precision
        self.register_buffer('_dummy', torch.empty(0), persistent=False)
        
    @property
    def dtype(self) -> torch.dtype:
        return self._dummy.dtype
        
    def _reshape_heads(self, x: torch.Tensor, is_key_value: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if is_key_value:
            x = x.view(batch, seq_len, self.gqa_groups, self.head_dim)
            x = x.transpose(1, 2)  # [batch, groups, seq, head_dim]
        else:
            x = x.view(batch, seq_len, self.n_heads, self.head_dim)
            x = x.transpose(1, 2)  # [batch, heads, seq, head_dim]
        return x
        
    def _compression_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, _, seq_len, _ = q.shape
        
        # Reshape inputs into blocks
        blocks = seq_len // self.block_size
        k_blocks = k.view(batch, -1, blocks, self.block_size, self.head_dim)
        v_blocks = v.view(batch, -1, blocks, self.block_size, self.head_dim)
        
        # Compress blocks
        k_compressed = self.compression_mlp(k_blocks)  # [batch, groups, blocks, head_dim]
        v_compressed = self.compression_mlp(v_blocks)  # [batch, groups, blocks, head_dim]
        
        if CUDA_KERNELS_AVAILABLE and q.is_cuda:
            # Use optimized CUDA kernel for attention computation
            block_indices = torch.arange(blocks, device=q.device).expand(batch, self.n_heads, -1)
            output = sparse_attention_cuda.sparse_attention_forward(
                q, k_compressed, v_compressed, block_indices,
                self.scale, self.block_size
            )
        else:
            # Fallback to PyTorch implementation
            attn = torch.matmul(q, k_compressed.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask[:, :, :, None] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v_compressed)
            
        return output
        
    def _selection_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, _, seq_len, _ = q.shape
        
        # Compute importance scores
        if CUDA_KERNELS_AVAILABLE and q.is_cuda:
            # Use optimized CUDA kernel for score computation
            scores = sparse_attention_cuda.sparse_attention_forward(
                q, k, v, None, self.scale, self.block_size
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :, None] == 0, float('-inf'))
            
        # Select top-n blocks (shared across GQA groups)
        block_scores = scores.view(batch, -1, seq_len // self.block_size, self.block_size).mean(dim=-1)
        _, top_indices = block_scores.topk(self.top_blocks, dim=-1)  # [batch, heads, top_blocks]
        
        if CUDA_KERNELS_AVAILABLE and q.is_cuda:
            # Use optimized CUDA kernel for attention with selected blocks
            output = sparse_attention_cuda.sparse_attention_forward(
                q, k, v, top_indices, self.scale, self.block_size
            )
        else:
            # Gather selected blocks
            k_selected = torch.gather(k.view(batch, -1, seq_len // self.block_size, self.block_size, self.head_dim),
                                    2, top_indices[..., None, None].expand(-1, -1, -1, self.block_size, self.head_dim))
            v_selected = torch.gather(v.view(batch, -1, seq_len // self.block_size, self.block_size, self.head_dim),
                                    2, top_indices[..., None, None].expand(-1, -1, -1, self.block_size, self.head_dim))
                                    
            # Compute attention with selected blocks
            k_selected = k_selected.view(batch, -1, self.top_blocks * self.block_size, self.head_dim)
            v_selected = v_selected.view(batch, -1, self.top_blocks * self.block_size, self.head_dim)
            
            attn = torch.matmul(q, k_selected.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v_selected)
            
        return output
        
    def _window_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, _, seq_len, _ = q.shape
        
        # Create sliding window attention mask
        window_mask = torch.ones(seq_len, seq_len, device=q.device)
        for i in range(seq_len):
            window_start = max(0, i - self.window_size // 2)
            window_end = min(seq_len, i + self.window_size // 2)
            window_mask[i, window_start:window_end] = 1
            
        if mask is not None:
            window_mask = window_mask & mask
            
        if CUDA_KERNELS_AVAILABLE and q.is_cuda:
            # Use optimized CUDA kernel for windowed attention
            window_indices = torch.arange(seq_len, device=q.device).expand(batch, self.n_heads, -1)
            output = sparse_attention_cuda.sparse_attention_forward(
                q, k, v, window_indices, self.scale, self.window_size
            )
        else:
            # Compute windowed attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(window_mask[:, :, None, None] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)
            
        return output
    
    def _quantize_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_fp8_kv_cache or not k.requires_grad:
            return k, v
            
        # Dynamic quantization to FP8
        with torch.no_grad():
            self.k_scale = torch.max(torch.abs(k)).clamp(min=1e-5)
            self.v_scale = torch.max(torch.abs(v)).clamp(min=1e-5)
            
        k_q = torch.quantize_per_tensor(k / self.k_scale, 1.0, 0, torch.float8_e4m3fn)
        v_q = torch.quantize_per_tensor(v / self.v_scale, 1.0, 0, torch.float8_e4m3fn)
        
        return k_q * self.k_scale, v_q * self.v_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Cast to computation dtype (typically float16 or bfloat16 in mixed precision)
        comp_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else hidden_states.dtype
        hidden_states = hidden_states.to(comp_dtype)
        
        # Use automatic mixed precision context
        with torch.cuda.amp.autocast(enabled=True):
            # Project queries, keys, and values
            q = self._reshape_heads(self.q_proj(hidden_states))
            k = self._reshape_heads(self.k_proj(hidden_states), is_key_value=True)
            v = self._reshape_heads(self.v_proj(hidden_states), is_key_value=True)
            
            # Quantize KV cache if enabled (using fp8 for compression path)
            if self.use_fp8_kv_cache and self.training:
                k, v = self._quantize_kv(k, v)
            
            # Use gradient checkpointing for memory efficiency
            if self.gradient_checkpointing and self.training:
                compression_out = torch.utils.checkpoint.checkpoint(
                    self._compression_branch, q, k, v, attention_mask,
                    use_reentrant=False  # More memory efficient
                )
                selection_out = torch.utils.checkpoint.checkpoint(
                    self._selection_branch, q, k, v, attention_mask,
                    use_reentrant=False
                )
                window_out = torch.utils.checkpoint.checkpoint(
                    self._window_branch, q, k, v, attention_mask,
                    use_reentrant=False
                )
            else:
                compression_out = self._compression_branch(q, k, v, attention_mask)
                selection_out = self._selection_branch(q, k, v, attention_mask)
                window_out = self._window_branch(q, k, v, attention_mask)
            
            # Compute gating weights with input in original precision
            gates = self.gate_mlp(hidden_states.to(self.dtype))  # [batch, seq, 3]
            g_cmp, g_slc, g_win = gates.to(comp_dtype).chunk(3, dim=-1)
            
            # Combine branch outputs with gating
            combined = (
                g_cmp.unsqueeze(1) * compression_out +
                g_slc.unsqueeze(1) * selection_out +
                g_win.unsqueeze(1) * window_out
            )
            
            # Reshape and project output
            combined = combined.transpose(1, 2).contiguous()
            combined = combined.view(batch_size, seq_length, self.d_model)
            
            # Final output projection
            return self.o_proj(combined) 