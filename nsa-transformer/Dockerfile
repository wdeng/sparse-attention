FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch and related packages
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    hydra-core==1.3.2 \
    wandb==0.16.0 \
    transformers==4.36.0 \
    datasets==2.15.0 \
    accelerate==0.25.0 \
    sentencepiece==0.1.99 \
    einops==0.7.0 \
    flash-attn==2.3.3 \
    triton==2.1.0 \
    pybind11==2.11.1 \
    ninja==1.11.1

# Set up working directory
WORKDIR /workspace/nsa-transformer

# Copy source code
COPY . .

# Build CUDA extensions
RUN cd kernels/sparse_attention && \
    python setup.py build_ext --inplace

# Set default command
CMD ["python", "-m", "torch.distributed.run", \
     "--nproc_per_node=8", \
     "training/trainer.py"] 