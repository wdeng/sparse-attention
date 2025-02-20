import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple, Dict
import math

class CosineSchedulerWithWarmup(LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Get learning rate based on current step."""
        step = self.last_epoch
        
        # Get base learning rates
        base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if step < self.warmup_steps:
            # Linear warmup
            scale = float(step) / float(max(1, self.warmup_steps))
            return [lr * scale for lr in base_lrs]
        else:
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scale between max and min learning rates
            return [
                self.min_lr + (lr - self.min_lr) * scale
                for lr in base_lrs
            ]

def create_optimizer_scheduler(
    model: torch.nn.Module,
    optimizer_config: Dict,
    scheduler_config: Dict,
) -> Tuple[torch.optim.Optimizer, LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )
    
    # Create scheduler
    scheduler = CosineSchedulerWithWarmup(
        optimizer=optimizer,
        warmup_steps=scheduler_config.warmup_steps,
        max_steps=scheduler_config.max_steps,
        min_lr=scheduler_config.min_lr,
    )
    
    return optimizer, scheduler 