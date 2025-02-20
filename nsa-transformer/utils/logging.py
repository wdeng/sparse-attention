import wandb
import torch
from typing import Dict, Any
from omegaconf import DictConfig
import os
import json
from datetime import datetime

class WandBLogger:
    """Weights & Biases logger with metric tracking."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Initialize W&B
        wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            config=self._flatten_config(config),
            name=self._get_run_name(),
        )
        
        # Setup logging directory
        self.log_dir = os.path.join(config.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metric history
        self.metric_history: Dict[str, list] = {}
        
    def _flatten_config(self, config: Any, parent_key: str = "") -> Dict:
        """Flatten nested config for W&B logging."""
        items = {}
        if isinstance(config, DictConfig):
            for key, value in config.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, (DictConfig, dict)):
                    items.update(self._flatten_config(value, new_key))
                else:
                    items[new_key] = value
        return items
        
    def _get_run_name(self) -> str:
        """Generate unique run name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model.get("name", "nsa")
        return f"{model_name}_{timestamp}"
        
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to W&B and local file."""
        # Update metric history
        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = []
            self.metric_history[name].append((step, value))
            
        # Log to W&B
        wandb.log(metrics, step=step)
        
        # Log to local file
        log_file = os.path.join(self.log_dir, "metrics.jsonl")
        with open(log_file, "a") as f:
            log_entry = {"step": step, **metrics}
            f.write(json.dumps(log_entry) + "\n")
            
    def log_model_graph(self, model: torch.nn.Module) -> None:
        """Log model graph to W&B."""
        wandb.watch(
            model,
            log="all",
            log_freq=self.config.logging.wandb.log_interval,
        )
        
    def log_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Log gradient statistics to W&B."""
        grad_norm = 0.0
        param_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                param_norm += param.data.norm(2).item() ** 2
                
                # Log layer-wise gradients
                wandb.log({
                    f"gradients/{name}_norm": param.grad.data.norm(2).item(),
                    f"parameters/{name}_norm": param.data.norm(2).item(),
                }, step=step)
                
        grad_norm = grad_norm ** 0.5
        param_norm = param_norm ** 0.5
        
        # Log global norms
        wandb.log({
            "gradients/global_norm": grad_norm,
            "parameters/global_norm": param_norm,
        }, step=step)
        
    def log_memory_stats(self, step: int) -> None:
        """Log GPU memory statistics to W&B."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            
            # Log memory stats
            wandb.log({
                "memory/allocated_gb": allocated,
                "memory/reserved_gb": reserved,
                "memory/max_allocated_gb": max_allocated,
            }, step=step)
            
    def finish(self) -> None:
        """Finish logging and save final metrics."""
        # Save metric history
        history_file = os.path.join(self.log_dir, "metric_history.json")
        with open(history_file, "w") as f:
            json.dump(self.metric_history, f, indent=2)
            
        # Finish W&B run
        wandb.finish() 