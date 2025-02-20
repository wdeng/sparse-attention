# NSA-Transformer: Neural Sparse Attention Transformer

A PyTorch implementation of the Neural Sparse Attention Transformer with distributed training support, memory optimizations, and mixture of experts.

## Features

- **Neural Sparse Attention**: Three-branch attention mechanism with compression, selection, and window attention
- **Mixture of Experts**: Dynamic routing with CPU offloading and load balancing
- **Memory Optimizations**:
  - FP8 KV cache quantization
  - Selective activation checkpointing
  - Expert CPU offloading
  - Optimized CUDA kernels
- **Distributed Training**:
  - Fully Sharded Data Parallel (FSDP)
  - Mixed precision training
  - Gradient accumulation
- **Data Processing**:
  - Streaming dataloader
  - Efficient sequence packing
  - Custom tokenizer with long context support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nsa-transformer.git
cd nsa-transformer
```

2. Create a conda environment:
```bash
conda create -n nsa python=3.10
conda activate nsa
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Build CUDA extensions:
```bash
cd nsa-transformer/kernels/sparse_attention
python setup.py build_ext --inplace
```

## Project Structure

```
nsa-transformer/
├── configs/                 # Hydra configuration files
├── data/                   # Data processing utilities
├── kernels/                # CUDA kernel implementations
├── models/                 # Model architecture
├── tests/                  # Unit tests
├── training/               # Training utilities
├── utils/                  # Helper functions
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### Training

1. Single GPU training:
```bash
python training/trainer.py
```

2. Multi-GPU training:
```bash
torchrun --nproc_per_node=8 training/trainer.py
```

3. Custom configuration:
```bash
python training/trainer.py model=large data.max_length=16384
```

### Docker

1. Build the container:
```bash
docker build -t nsa-transformer .
```

2. Run training:
```bash
docker run --gpus all nsa-transformer
```

## Configuration

The project uses Hydra for configuration management. Main configuration files:

- `configs/config.yaml`: Root configuration
- `configs/model/base.yaml`: Model architecture
- `configs/training/base.yaml`: Training settings
- `configs/memory/base.yaml`: Memory optimizations
- `configs/data/base.yaml`: Data processing
- `configs/logging/base.yaml`: Logging and monitoring

### Example Configuration

```yaml
# Train a large model with increased context length
python training/trainer.py \
    model=large \
    model.max_position_embeddings=32768 \
    training.batch_size=16 \
    memory.kv_cache.quantization=fp8
```

## Memory Optimizations

1. Enable KV cache quantization:
```yaml
memory:
  kv_cache:
    quantization: fp8
    threshold: 512
```

2. Configure expert offloading:
```yaml
memory:
  expert_offloading:
    enabled: true
    pin_memory: true
```

3. Optimize CUDA kernels:
```yaml
memory:
  cuda_kernels:
    enable_hbm_optimizations: true
    enable_bank_conflict_avoidance: true
```

## Monitoring

1. Setup Weights & Biases:
```yaml
logging:
  wandb:
    project: "your-project"
    entity: "your-username"
```

2. Enable profiling:
```yaml
logging:
  profiling:
    enabled: true
    trace_memory: true
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_attention.py
pytest tests/test_moe.py
pytest tests/test_training.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nsa-transformer,
  author = {Your Name},
  title = {NSA-Transformer: Neural Sparse Attention Transformer},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/nsa-transformer}
}
``` 