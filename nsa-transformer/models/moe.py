import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ExpertFFN(nn.Module):
    """Expert feed-forward network with larger capacity than standard FFN."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, 8 * d_model)  # Wider than standard FFN
        self.w2 = nn.Linear(8 * d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))

class MoERouter(nn.Module):
    """Router network that assigns tokens to experts."""
    
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        
    def forward(
        self,
        x: torch.Tensor,
        n_active: int,
        capacity: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get router scores and probabilities
        router_logits = self.router(x)  # [batch, seq, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        scores, expert_indices = torch.topk(router_probs, k=n_active, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)  # Normalize selected probabilities
        
        # Create dispatch tensors
        batch_size, seq_len, _ = x.shape
        capacity_size = min(capacity, seq_len)
        
        # Create binary gates and combine with scores
        gates = torch.zeros_like(router_probs)
        gates.scatter_(-1, expert_indices, scores)
        
        # Compute load balancing auxiliary loss
        # Penalize when experts are assigned more tokens than capacity
        expert_counts = gates.sum(dim=(0, 1))
        load_loss = torch.mean((expert_counts / expert_counts.sum() * self.n_experts - 1) ** 2)
        
        return gates, expert_indices, load_loss

class MoELayer(nn.Module):
    """Mixture of Experts layer with token routing and load balancing."""
    
    def __init__(
        self,
        d_model: int,
        n_experts: int = 72,
        n_active_experts: int = 6,
        expert_capacity: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active_experts
        self.capacity = expert_capacity
        
        # Create experts (including 2 shared experts)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, dropout)
            for _ in range(n_experts)
        ])
        self.shared_experts = nn.ModuleList([
            ExpertFFN(d_model, dropout)
            for _ in range(2)  # 2 shared experts as in DeepSeek
        ])
        
        # Router network
        self.router = MoERouter(d_model, n_experts + 2)  # +2 for shared experts
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        
        # Get routing probabilities and expert assignments
        gates, expert_indices, load_loss = self.router(x, self.n_active, self.capacity)
        
        # Process tokens through experts
        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = gates[:, :, i].bool()
            if not expert_mask.any():
                continue
                
            # Process tokens through expert
            expert_input = x[expert_mask]
            expert_output = expert(expert_input)
            final_output[expert_mask] += expert_output * gates[:, :, i][expert_mask, None]
            
        # Process through shared experts
        for i, expert in enumerate(self.shared_experts):
            expert_idx = self.n_experts + i
            expert_mask = gates[:, :, expert_idx].bool()
            if not expert_mask.any():
                continue
                
            expert_input = x[expert_mask]
            expert_output = expert(expert_input)
            final_output[expert_mask] += expert_output * gates[:, :, expert_idx][expert_mask, None]
            
        return final_output, load_loss 