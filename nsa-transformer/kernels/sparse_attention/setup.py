from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA compute capability of the current device
def get_cuda_arch():
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f'sm_{major}{minor}'
    return 'sm_70'  # Default to Volta

cuda_arch = get_cuda_arch()

# CUDA compilation flags for optimization
CUDA_FLAGS = [
    '-O3',                      # Highest optimization level
    '--use_fast_math',         # Fast math operations
    '-Xptxas=-v',              # Verbose PTX assembly
    '-Xcompiler=-O3',          # Host code optimization
    '-Xcompiler=-march=native', # CPU architecture specific optimizations
    '--threads=4',             # Parallel compilation
    '--maxrregcount=128',      # Limit register usage for better occupancy
    f'--gpu-architecture={cuda_arch}',  # Target architecture
    '--ftz=true',              # Flush denormals to zero
    '--prec-div=false',        # Fast division
    '--prec-sqrt=false',       # Fast square root
    '--fmad=true',             # Fused multiply-add
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
    '/usr/local/cuda/include'  # CUDA include directory
]

# Setup the extension
setup(
    name='sparse_attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='sparse_attention_cuda',
            sources=[os.path.join(os.path.dirname(__file__), src) for src in sources],
            extra_compile_args={
                'cxx': ['-O3', '-march=native', '-fopenmp'],
                'nvcc': CUDA_FLAGS
            },
            include_dirs=include_dirs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 