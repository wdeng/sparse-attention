#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "kernel_utils.cuh"

// Forward declarations of kernel functions
template<typename scalar_t>
__global__ void compute_block_attention_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    scalar_t* __restrict__ scores,
    const int* __restrict__ block_indices,
    const int seq_length,
    const int head_dim,
    const int block_size,
    const float scale
);

template<typename scalar_t>
__global__ void compute_attention_output_kernel(
    const scalar_t* __restrict__ attn_weights,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_indices,
    const int seq_length,
    const int head_dim,
    const int block_size
);

// Main forward function called from Python
torch::Tensor sparse_attention_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const float scale,
    const int block_size
) {
    PROFILE_KERNEL("sparse_attention_forward");
    
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_length = query.size(2);
    const auto head_dim = query.size(3);
    
    // Allocate output tensors in same dtype as input
    auto scores = torch::zeros({batch_size, num_heads, seq_length, seq_length},
                             query.options());
    auto output = torch::zeros({batch_size, num_heads, seq_length, head_dim},
                             query.options());
    
    // Calculate grid and block dimensions
    const dim3 grid = get_grid_dim(batch_size, num_heads, seq_length, block_size);
    const dim3 block = get_block_dim(block_size);
    
    // Calculate shared memory size with padding for different precisions
    size_t elem_size = query.element_size();
    const int shared_mem_size = 2 * block.x * head_dim * elem_size + 
                               (elem_size <= 2 ? 2 * HBM_BANK_SIZE : HBM_BANK_SIZE);  // Extra padding for FP16/BF16
    
    // Launch kernels with type dispatch
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                   query.scalar_type(), "sparse_attention_forward", ([&] {
        compute_block_attention_scores_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            scores.data_ptr<scalar_t>(),
            block_indices.defined() ? block_indices.data_ptr<int>() : nullptr,
            seq_length,
            head_dim,
            block_size,
            scale
        );
        
        compute_attention_output_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            scores.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_indices.defined() ? block_indices.data_ptr<int>() : nullptr,
            seq_length,
            head_dim,
            block_size
        );
    }));
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA kernel error");
    }
    
    return output;
} 