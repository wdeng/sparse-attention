<blueprint>
  <directory_structure>
    nsa-transformer/
    ├── configs/                 # Hydra configuration files
    │   ├── model/               # Model architecture configurations
    │   ├── training/            # Training hyperparameters
    │   └── experiment.yaml      # Main experiment configuration
    ├── data/                    # Data processing utilities
    │   ├── dataloader.py        # Streaming data loader for long sequences
    │   └── tokenizer.py         # Custom tokenizer handling 64k+ contexts
    ├── kernels/                 # CUDA kernel implementations
    │   ├── sparse_attention/    # NSA-specific CUDA kernels
    │   │   ├── attention.cpp    # PyBind11 bindings for CUDA kernels
    │   │   └── kernel_utils.cuh # Shared CUDA utilities
    ├── models/                  # Model architecture components
    │   ├── attention.py         # NSA attention layer implementation
    │   ├── moe.py               # Mixture-of-Experts implementation
    │   └── transformer.py       # Main transformer architecture
    ├── tests/                   # pytest unit tests
    │   ├── test_attention.py    # NSA attention layer tests
    │   └── test_kernels.py      # CUDA kernel validation
    ├── training/                # Training loop utilities
    │   ├── optim.py             # Custom optimizer configuration
    │   └── trainer.py           # Distributed training logic
    ├── utils/                   # Utility functions
    │   ├── logging.py           # W&B integration and metric tracking
    │   └── memory.py            # Memory optimization utilities
    └── Dockerfile               # CUDA 12.x + PyTorch 2.2 environment
  </directory_structure>

  <core_components>
    ## NSA Attention Layer Implementation ##
    Class NSAttention(nn.Module):
      Implements three parallel attention branches:
      1. Token Compression: 
         - Input sequence divided into blocks of size l=32
         - Each block processed by MLP φ: R^{l×d} → R^d with positional encodings
         - Compression stride d=16 creates overlapping blocks
        
      2. Blockwise Selection:
         - Compute importance scores using compression branch attention weights
         - Select top-n=16 blocks (size l'=64) via differentiable ranking
         - Shared selection across GQA groups using Equation (10)
        
      3. Sliding Window:
         - Fixed window size w=512 tokens
         - Implemented via causal mask + FlashAttention-2
        
      Gate Mechanism:
        - Learnable gating weights per branch: g^cmp, g^slc, g^win
        - MLP: h → Linear(2560 → 3) → Sigmoid
        - Final output: o_t = Σ g^c·Attn(q_t, K^c, V^c)

    ## CUDA Kernels ##
    1. Group-Centric Sparse Attention:
       - Implements Figure 3 logic using ATen API
       - Key optimizations:
         a) Load all queries in GQA group (h=16 heads) simultaneously
         b) Fetch KV blocks contiguously from HBM using shared indices
         c) Block size B_k=64 to match l' parameter
         d) Outer loop parallelized via Triton-style grid scheduler

    2. Blockwise Top-n Selection:
       - Custom kernel for Equation (11) ranking
       - Avoids full sort via bitonic sort networks
       - Maintains gradients through ranking operation

    ## MoE Implementation ##
    - DeepSeekMoE architecture: 72 experts (6 active) + 2 shared experts
    - Expert parallelism using Fully Sharded Data Parallel
    - Gating network: Router(Z) = Softmax(Linear(hidden, 74))
  </core_components>

  <training_setup>
    ## Hyperparameters ##
    - Architecture: 30 layers, d_model=2560, GQA groups=4
    - Optimization: AdamW (β1=0.9, β2=0.95)
    - Learning Rate: Peak 2e-4 with cosine decay to 2e-5
    - Batch Size: 4M tokens/batch (512 sequences × 8k tokens)
    - Context Scaling: YaRN up to 64k via rope_theta=1e6
    - Regularization: 0.1 dropout, 0.01 weight decay

    ## Distributed Training ##
    - 8x A100 nodes with FSDP
    - Activation Checkpointing: All attention layers
    - Precision: bfloat16 AMP with dynamic loss scaling

    ## Data Pipeline ##
    - Streaming dataset from HuggingFace Hub
    - Sequence packing for variable-length documents
    - 4:1 pretraining (8k) → fine-tuning (32k) ratio
    - Vocabulary: 128k tokens with special <fill> for long docs
  </training_setup>

  <optimizations>
    1. Memory Optimization:
       - KV Cache Quantization: FP8 for >16k contexts
       - Zero-3 Offloading for expert parameters
       - Selective Activation Checkpointing

    2. Kernel Fusion:
       - Fuse softmax + top-n selection + gating
       - Combine QKV projections for all branches
       - FlashAttention-2 integration for window branch

    3. CUDA Extensions:
       - Blockwise MLP Compression (φ)
       - Grouped Sparse Attention Backward Pass
       - MoE Expert Dispatch Kernels

    4. Profiling:
       - Nsight Systems for kernel performance
       - PyTorch Profiler with W&B integration
       - Memory bandwidth utilization metrics
  </optimizations>

  <assumptions>
    1. Compression MLP φ uses 2-layer GeLU network
    2. Sliding window implements "local + fixed sinks" pattern
    3. GQA group size=4 based on paper's configuration
    4. Token selection uses straight-through estimator
    5. Initial 1000 steps use full attention warmup
    6. YaRN scaling uses base implementation
    7. Expert balancing uses load loss from DeepSeekMoE
  </assumptions>
</blueprint>