#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "kernel_utils.cuh"

// Backward kernels for gradient computation
template<typename scalar_t>
__global__ void compute_attention_grad_output_kernel(
    const scalar_t* __restrict__ grad_output,    // [batch, heads, seq_len, head_dim]
    const scalar_t* __restrict__ value,          // [batch, heads, seq_len, head_dim]
    const scalar_t* __restrict__ attn_weights,   // [batch, heads, seq_len, seq_len]
    scalar_t* __restrict__ grad_query,           // [batch, heads, seq_len, head_dim]
    scalar_t* __restrict__ grad_key,             // [batch, heads, seq_len, head_dim]
    scalar_t* __restrict__ grad_value,           // [batch, heads, seq_len, head_dim]
    const int* __restrict__ block_indices,       // [batch, heads, num_blocks]
    const int seq_length,
    const int head_dim,
    const int block_size,
    const float scale
) {
    // Shared memory allocation with padding for bank conflict avoidance
    extern __shared__ char shared_memory[];
    scalar_t* shared_grad = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_val = shared_grad + blockDim.x * head_dim + HBM_BANK_SIZE / sizeof(scalar_t);
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_idx = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;
    
    const int q_block_start = block_idx * block_size;
    const int batch_offset = (batch_idx * gridDim.y + head_idx) * seq_length;
    
    // Load gradient block into shared memory
    if (q_block_start + thread_idx < seq_length) {
        load_to_shared_memory_vectorized(
            shared_grad + thread_idx * head_dim,
            grad_output + (batch_offset + q_block_start + thread_idx) * head_dim,
            0, 1, head_dim
        );
    }
    
    __syncthreads();
    
    // Process blocks with warp-level parallelism
    for (int k_block_idx = warp_idx; k_block_idx < gridDim.z; k_block_idx += blockDim.x / WARP_SIZE) {
        const int selected_block = block_indices ? 
            block_indices[(batch_idx * gridDim.y + head_idx) * gridDim.z + k_block_idx] :
            k_block_idx;
        const int k_block_start = selected_block * block_size;
        
        // Load value block into shared memory
        if (k_block_start + lane_idx < seq_length) {
            load_to_shared_memory_vectorized(
                shared_val + lane_idx * head_dim,
                value + (batch_offset + k_block_start + lane_idx) * head_dim,
                0, 1, head_dim
            );
        }
        
        __syncwarp();
        
        // Compute gradients with SIMD instructions
        if (q_block_start + thread_idx < seq_length) {
            scalar_t grad_vec[4];  // Load 4 elements at once
            #pragma unroll
            for (int d = 0; d < head_dim; d += 4) {
                // Load gradient vector
                *reinterpret_cast<float4*>(grad_vec) = *reinterpret_cast<const float4*>(
                    shared_grad + thread_idx * head_dim + d
                );
                
                for (int k_offset = 0; k_offset < block_size && k_block_start + k_offset < seq_length; k_offset++) {
                    const scalar_t weight = attn_weights[
                        (batch_offset + q_block_start + thread_idx) * seq_length + 
                        k_block_start + k_offset
                    ];
                    
                    scalar_t val_vec[4];
                    // Load value vector
                    *reinterpret_cast<float4*>(val_vec) = *reinterpret_cast<const float4*>(
                        shared_val + k_offset * head_dim + d
                    );
                    
                    // Compute gradients
                    const int grad_idx = (batch_offset + k_block_start + k_offset) * head_dim + d;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        atomicAdd(&grad_value[grad_idx + i], weight * grad_vec[i]);
                        atomicAdd(&grad_key[grad_idx + i], weight * grad_vec[i] * scale);
                    }
                }
            }
        }
        
        __syncthreads();
    }
}

// Main backward function called from Python
std::vector<torch::Tensor> sparse_attention_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const torch::Tensor& attn_weights,
    const float scale,
    const int block_size
) {
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_length = query.size(2);
    const auto head_dim = query.size(3);
    
    // Allocate gradient tensors
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);
    auto grad_value = torch::zeros_like(value);
    
    // Calculate grid and block dimensions
    const dim3 grid = get_grid_dim(batch_size, num_heads, seq_length, block_size);
    const dim3 block = get_block_dim(block_size);
    
    // Calculate shared memory size
    const int shared_mem_size = 2 * block.x * head_dim * sizeof(float) + 
                               HBM_BANK_SIZE;  // Extra padding for bank conflicts
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "sparse_attention_backward", ([&] {
        compute_attention_grad_output_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            grad_output.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            attn_weights.data_ptr<scalar_t>(),
            grad_query.data_ptr<scalar_t>(),
            grad_key.data_ptr<scalar_t>(),
            grad_value.data_ptr<scalar_t>(),
            block_indices.defined() ? block_indices.data_ptr<int>() : nullptr,
            seq_length,
            head_dim,
            block_size,
            scale
        );
    }));
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA kernel error");
    }
    
    return {grad_query, grad_key, grad_value};
} 