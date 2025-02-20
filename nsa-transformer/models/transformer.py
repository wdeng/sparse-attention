import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import NSAttention
from .moe import MoELayer

class NSATransformerBlock(nn.Module):
    """Transformer block with NSA attention and optional MoE FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 32,
        compression_stride: int = 16,
        top_blocks: int = 16,
        window_size: int = 512,
        gqa_groups: int = 4,
        dropout: float = 0.1,
        use_moe: bool = False,
        n_experts: int = 72,
        n_active_experts: int = 6,
        expert_capacity: int = 128,
    ):
        super().__init__()
        self.use_moe = use_moe
        
        # Layer norm and attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = NSAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            compression_stride=compression_stride,
            top_blocks=top_blocks,
            window_size=window_size,
            gqa_groups=gqa_groups,
            dropout=dropout,
        )
        
        # FFN or MoE layer
        self.norm2 = nn.LayerNorm(d_model)
        if use_moe:
            self.ffn = MoELayer(
                d_model=d_model,
                n_experts=n_experts,
                n_active_experts=n_active_experts,
                expert_capacity=expert_capacity,
                dropout=dropout,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attention_mask)
        x = x + residual
        
        # FFN block
        residual = x
        x = self.norm2(x)
        if self.use_moe:
            x, aux_loss = self.ffn(x)
        else:
            x = self.ffn(x)
            aux_loss = None
        x = x + residual
        
        return x, aux_loss

class NSATransformer(nn.Module):
    """Neural Sparse Attention Transformer with optional MoE layers."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 2560,
        n_layers: int = 30,
        n_heads: int = 32,
        max_seq_length: int = 32768,
        dropout: float = 0.1,
        block_size: int = 32,
        compression_stride: int = 16,
        top_blocks: int = 16,
        window_size: int = 512,
        gqa_groups: int = 4,
        moe_layers: Optional[list[int]] = None,
        n_experts: int = 72,
        n_active_experts: int = 6,
        expert_capacity: int = 128,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        
        # Token embeddings and position encoding
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NSATransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                block_size=block_size,
                compression_stride=compression_stride,
                top_blocks=top_blocks,
                window_size=window_size,
                gqa_groups=gqa_groups,
                dropout=dropout,
                use_moe=(i in (moe_layers or [])),
                n_experts=n_experts,
                n_active_experts=n_active_experts,
                expert_capacity=expert_capacity,
            )
            for i in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_input_embeddings(self) -> nn.Module:
        return self.tok_emb
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = input_ids.size()
        
        # Generate position IDs and attention mask if not provided
        pos_ids = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)
            
        # Get embeddings
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(pos_ids)
        x = self.dropout(tok_emb + pos_emb)
        
        # Process through transformer layers
        aux_losses = []
        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
                
        # Final norm and head
        x = self.norm(x)
        logits = self.head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.pad_token_id,
            )
            if aux_losses:
                loss = loss + sum(aux_losses)
                
        return logits, loss 