import torch
import torch.nn as nn
from typing import Dict, Optional
from omegaconf import DictConfig
import torch.distributed as dist

def setup_memory_optimizations(
    model: nn.Module,
    config: DictConfig,
    local_rank: int,
) -> None:
    """Setup memory optimizations for the model."""
    # Setup KV cache quantization
    if config.kv_cache.quantization == "fp8":
        quantize_kv_cache(
            model=model,
            threshold=config.kv_cache.threshold,
            local_rank=local_rank,
        )
        
    # Setup activation checkpointing
    if config.activation_checkpointing.granularity == "selective":
        setup_selective_activation_checkpointing(
            model=model,
            checkpoint_every_n_layers=config.activation_checkpointing.checkpoint_every_n_layers,
        )

def quantize_kv_cache(
    model: nn.Module,
    threshold: int,
    local_rank: int,
) -> None:
    """Quantize KV cache to FP8 for sequences longer than threshold."""
    
    class KVCacheQuantizer(torch.autograd.Function):
        """Custom autograd function for FP8 quantization."""
        
        @staticmethod
        def forward(ctx, x: torch.Tensor, threshold: int) -> torch.Tensor:
            ctx.threshold = threshold
            ctx.save_for_backward(x)
            
            if x.size(1) > threshold:
                # Quantize to FP8
                scale = x.abs().max() / 127
                x_quant = torch.round(x / scale).clamp(-127, 127)
                x = x_quant * scale
                
            return x
            
        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            x, = ctx.saved_tensors
            
            if x.size(1) > ctx.threshold:
                # Dequantize gradients
                scale = grad_output.abs().max() / 127
                grad_quant = torch.round(grad_output / scale).clamp(-127, 127)
                grad_output = grad_quant * scale
                
            return grad_output, None
            
    # Apply quantization to KV cache in attention layers
    def apply_kv_quantization(module: nn.Module):
        if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            # Store original forward
            original_forward = module.forward
            
            # Create new forward with quantization
            def forward_with_quantization(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                if isinstance(outputs, tuple):
                    # Quantize KV cache
                    k, v = outputs[1], outputs[2]
                    k = KVCacheQuantizer.apply(k, threshold)
                    v = KVCacheQuantizer.apply(v, threshold)
                    outputs = (outputs[0], k, v, *outputs[3:])
                return outputs
                
            # Replace forward method
            module.forward = forward_with_quantization
            
    model.apply(apply_kv_quantization)
    
    if local_rank == 0:
        print(f"Enabled FP8 KV cache quantization for sequences > {threshold}")

def setup_selective_activation_checkpointing(
    model: nn.Module,
    checkpoint_every_n_layers: int,
) -> None:
    """Setup selective activation checkpointing for transformer layers."""
    from torch.utils.checkpoint import checkpoint
    
    def apply_checkpointing(module: nn.Module, layer_idx: int):
        if hasattr(module, "forward") and layer_idx % checkpoint_every_n_layers == 0:
            # Store original forward
            original_forward = module.forward
            
            # Create checkpointed forward
            def forward_with_checkpoint(*args, **kwargs):
                def custom_forward(*inputs):
                    return original_forward(*inputs, **kwargs)
                return checkpoint(custom_forward, *args)
                
            # Replace forward method
            module.forward = forward_with_checkpoint
            
    # Apply checkpointing to every nth layer
    for i, module in enumerate(model.modules()):
        apply_checkpointing(module, i)

def setup_expert_offloading(
    model: nn.Module,
    config: DictConfig,
) -> None:
    """Setup CPU offloading for MoE experts."""
    if not config.enabled:
        return
        
    def offload_experts(module: nn.Module):
        if hasattr(module, "experts"):
            # Move experts to CPU
            for expert in module.experts:
                expert.to("cpu")
                if config.pin_memory:
                    for param in expert.parameters():
                        param.pin_memory()
                        
            # Store original forward
            original_forward = module.forward
            
            # Create forward with expert offloading
            def forward_with_offloading(*args, **kwargs):
                # Move active experts to GPU
                active_experts = kwargs.get("active_experts", None)
                if active_experts is not None:
                    for idx in active_experts:
                        module.experts[idx].to(args[0].device)
                        
                outputs = original_forward(*args, **kwargs)
                
                # Move experts back to CPU
                if active_experts is not None:
                    for idx in active_experts:
                        module.experts[idx].to("cpu")
                        
                return outputs
                
            # Replace forward method
            module.forward = forward_with_offloading
            
    model.apply(offload_experts) 