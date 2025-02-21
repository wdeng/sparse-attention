from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# Get CUDA compute capability of the current device
def get_cuda_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f'sm_{major}{minor}'
    return 'sm_80'  # Default to Ampere

cuda_arch = get_cuda_arch()

# CUDA compilation flags for optimization
CUDA_FLAGS = [
    '-O3',                      # Highest optimization level
    '--use_fast_math',         # Fast math operations
    '--threads=4',             # Parallel compilation
    '--maxrregcount=128',      # Limit register usage for better occupancy
    f'--gpu-architecture={cuda_arch}',  # Target architecture
    '--ftz=true',              # Flush denormals to zero
    '--prec-div=false',        # Fast division
    '--prec-sqrt=false',       # Fast square root
    '--fmad=true',             # Fused multiply-add
    '--use-tensor-cores',      # Enable tensor cores
    '--ptxas-options=-v',      # Verbose PTX assembly
    '--ptxas-options=-dlcm=ca',  # L1 cache optimization
    '--default-stream=per-thread',  # Thread-local streams
    '--restrict',              # Enable restrict keyword
    '--extra-device-vectorization',  # Additional vectorization
    '--Wno-deprecated-gpu-targets',  # Suppress deprecation warnings
]

# Host compiler flags
CXX_FLAGS = [
    '-O3',                # High optimization
    '-march=native',      # CPU architecture specific optimizations
    '-fopenmp',          # OpenMP support
    '-ffast-math',       # Fast math operations
    '-fno-finite-math-only',  # Allow non-finite math
]

# Source files
sources = [
    'attention.cpp',
    'attention_forward.cu',
    'attention_backward.cu'
]

# Include directories
include_dirs = [
    os.path.dirname(os.path.abspath(__file__)),
    '/usr/local/cuda/include',  # CUDA include directory
    os.path.dirname(torch.__file__),  # PyTorch include directory
]

# Library directories
library_dirs = [
    '/usr/local/cuda/lib64',  # CUDA library directory
]

# Setup the extension
setup(
    name='sparse_attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='sparse_attention_cuda',
            sources=[os.path.join(os.path.dirname(__file__), src) for src in sources],
            extra_compile_args={
                'cxx': CXX_FLAGS,
                'nvcc': CUDA_FLAGS
            },
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['cudart', 'cublas', 'cusparse']  # Required CUDA libraries
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 