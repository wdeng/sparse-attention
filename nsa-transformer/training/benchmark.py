import torch
import time
import hydra
from omegaconf import DictConfig
import numpy as np
from typing import Dict, List
import json
import os

from models.transformer import NSATransformer
from utils.memory import setup_memory_optimizations

def generate_dummy_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate dummy batch for benchmarking."""
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
        "attention_mask": torch.ones(batch_size, seq_length, device=device),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
    }

def measure_memory(device: torch.device) -> Dict[str, float]:
    """Measure GPU memory usage."""
    if device.type == "cuda":
        return {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        }
    return {}

def benchmark_forward(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    n_steps: int,
) -> Dict[str, float]:
    """Benchmark forward pass."""
    timings = []
    
    # Warmup
    for _ in range(3):
        _ = model(**batch)
    torch.cuda.synchronize()
    
    # Measure
    for _ in range(n_steps):
        start = time.perf_counter()
        _ = model(**batch)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        
    return {
        "forward_mean_ms": np.mean(timings) * 1000,
        "forward_std_ms": np.std(timings) * 1000,
    }

def benchmark_forward_backward(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    n_steps: int,
) -> Dict[str, float]:
    """Benchmark forward and backward pass."""
    timings = []
    
    # Warmup
    for _ in range(3):
        outputs = model(**batch)
        loss = outputs[1]
        loss.backward()
    torch.cuda.synchronize()
    
    # Measure
    for _ in range(n_steps):
        start = time.perf_counter()
        outputs = model(**batch)
        loss = outputs[1]
        loss.backward()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        
    return {
        "forward_backward_mean_ms": np.mean(timings) * 1000,
        "forward_backward_std_ms": np.std(timings) * 1000,
    }

def benchmark_memory_scaling(
    model: torch.nn.Module,
    batch_size: int,
    seq_lengths: List[int],
    vocab_size: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Benchmark memory scaling with sequence length."""
    memory_usage = {
        "seq_lengths": seq_lengths,
        "allocated_gb": [],
        "reserved_gb": [],
    }
    
    for seq_length in seq_lengths:
        # Clear cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        # Generate batch and run forward pass
        batch = generate_dummy_batch(batch_size, seq_length, vocab_size, device)
        _ = model(**batch)
        
        # Measure memory
        memory = measure_memory(device)
        memory_usage["allocated_gb"].append(memory.get("allocated_gb", 0))
        memory_usage["reserved_gb"].append(memory.get("reserved_gb", 0))
        
    return memory_usage

@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = hydra.utils.instantiate(config.model).to(device)
    model.eval()
    
    # Setup memory optimizations
    setup_memory_optimizations(model, config.memory, 0)
    
    # Generate dummy batch
    batch = generate_dummy_batch(
        batch_size=config.training.batch_size,
        seq_length=config.data.max_length,
        vocab_size=config.model.vocab_size,
        device=device,
    )
    
    # Run benchmarks
    results = {}
    
    # Forward pass benchmark
    results.update(benchmark_forward(
        model=model,
        batch=batch,
        n_steps=config.profiling.benchmark_steps,
    ))
    
    # Forward + backward pass benchmark
    results.update(benchmark_forward_backward(
        model=model,
        batch=batch,
        n_steps=config.profiling.benchmark_steps,
    ))
    
    # Memory scaling benchmark
    results["memory_scaling"] = benchmark_memory_scaling(
        model=model,
        batch_size=config.training.batch_size,
        seq_lengths=[1024, 2048, 4096, 8192, 16384, 32768],
        vocab_size=config.model.vocab_size,
        device=device,
    )
    
    # Add configuration to results
    results["config"] = {
        "model_size": config.model.d_model,
        "n_layers": config.model.n_layers,
        "batch_size": config.training.batch_size,
        "seq_length": config.data.max_length,
    }
    
    # Save results
    output_file = os.path.join(config.output_dir, "benchmark_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main() 