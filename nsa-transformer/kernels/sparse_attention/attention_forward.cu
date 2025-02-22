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
    const float scale,
    const bool spatial_mode = false,
    const int height = 0,
    const int width = 0,
    const scalar_t* __restrict__ rel_pos_h = nullptr,
    const scalar_t* __restrict__ rel_pos_w = nullptr
);

template<typename scalar_t>
__global__ void compute_attention_output_kernel(
    const scalar_t* __restrict__ attn_weights,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    const int* __restrict__ block_indices,
    const int seq_length,
    const int head_dim,
    const int block_size,
    const bool spatial_mode = false,
    const int height = 0,
    const int width = 0
);

// Main forward function called from Python
torch::Tensor sparse_attention_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const float scale,
    const int block_size,
    const bool spatial_mode = false,
    const c10::optional<std::tuple<int, int>>& image_size = c10::nullopt,
    const c10::optional<std::pair<torch::Tensor, torch::Tensor>>& rel_pos = c10::nullopt
) {
    PROFILE_KERNEL("sparse_attention_forward");
    
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_length = query.size(2);
    const auto head_dim = query.size(3);
    
    int height = 0, width = 0;
    if (spatial_mode) {
        TORCH_CHECK(image_size.has_value(), "image_size must be provided for spatial mode");
        std::tie(height, width) = image_size.value();
        TORCH_CHECK(seq_length == height * width, 
                   "sequence length must match image dimensions");
    }
    
    // Allocate output tensors in same dtype as input
    auto scores = torch::zeros({batch_size, num_heads, seq_length, seq_length},
                             query.options());
    auto output = torch::zeros({batch_size, num_heads, seq_length, head_dim},
                             query.options());
    
    // Calculate grid and block dimensions
    const dim3 grid = spatial_mode ? 
        get_grid_dim_2d(batch_size, num_heads, height, width, block_size) :
        get_grid_dim(batch_size, num_heads, seq_length, block_size);
        
    const dim3 block = spatial_mode ?
        get_block_dim_2d(block_size) :
        get_block_dim(block_size);
    
    // Calculate shared memory size with padding for different precisions
    size_t elem_size = query.element_size();
    const int shared_mem_size = spatial_mode ?
        (2 * block_size * block_size * head_dim * elem_size + 
         2 * HBM_BANK_SIZE) :  // Extra padding for 2D blocks
        (2 * block.x * head_dim * elem_size + 
         (elem_size <= 2 ? 2 * HBM_BANK_SIZE : HBM_BANK_SIZE));
    
    // Launch kernels with type dispatch
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                   query.scalar_type(), "sparse_attention_forward", ([&] {
        const scalar_t* rel_pos_h_ptr = nullptr;
        const scalar_t* rel_pos_w_ptr = nullptr;
        if (spatial_mode && rel_pos.has_value()) {
            rel_pos_h_ptr = rel_pos.value().first.data_ptr<scalar_t>();
            rel_pos_w_ptr = rel_pos.value().second.data_ptr<scalar_t>();
        }
        
        compute_block_attention_scores_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            scores.data_ptr<scalar_t>(),
            block_indices.defined() ? block_indices.data_ptr<int>() : nullptr,
            seq_length,
            head_dim,
            block_size,
            scale,
            spatial_mode,
            height,
            width,
            rel_pos_h_ptr,
            rel_pos_w_ptr
        );
        
        compute_attention_output_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            scores.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            block_indices.defined() ? block_indices.data_ptr<int>() : nullptr,
            seq_length,
            head_dim,
            block_size,
            spatial_mode,
            height,
            width
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

// Kernel implementation for block attention scores with 2D support
template<typename scalar_t>
__global__ void compute_block_attention_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    scalar_t* __restrict__ scores,
    const int* __restrict__ block_indices,
    const int seq_length,
    const int head_dim,
    const int block_size,
    const float scale,
    const bool spatial_mode,
    const int height,
    const int width,
    const scalar_t* __restrict__ rel_pos_h,
    const scalar_t* __restrict__ rel_pos_w
) {
    // Shared memory allocation with padding for bank conflict avoidance
    extern __shared__ char shared_memory[];
    scalar_t* shared_q = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_k = shared_q + (spatial_mode ? block_size * block_size * head_dim : 
                                    blockDim.x * head_dim) + 
                         HBM_BANK_SIZE / sizeof(scalar_t);
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_idx = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;
    
    if (spatial_mode) {
        // 2D spatial attention
        const int blocks_w = width / block_size;
        const BlockCoords q_block = {
            block_idx / blocks_w,
            block_idx % blocks_w,
            block_size
        };
        
        // Load query block
        load_2d_block_to_shared(
            shared_q,
            query + (batch_idx * gridDim.y + head_idx) * seq_length * head_dim,
            q_block,
            height,
            width,
            head_dim
        );
        
        __syncthreads();
        
        // Process key blocks with memory coalescing
        for (int k_block_idx = warp_idx; k_block_idx < gridDim.z; k_block_idx += blockDim.x / WARP_SIZE) {
            const int selected_block = block_indices ? 
                block_indices[(batch_idx * gridDim.y + head_idx) * gridDim.z + k_block_idx] :
                k_block_idx;
                
            const BlockCoords k_block = {
                selected_block / blocks_w,
                selected_block % blocks_w,
                block_size
            };
            
            // Load key block
            load_2d_block_to_shared(
                shared_k,
                key + (batch_idx * gridDim.y + head_idx) * seq_length * head_dim,
                k_block,
                height,
                width,
                head_dim
            );
            
            __syncwarp();
            
            // Compute attention scores with relative position bias
            for (int q_idx = lane_idx; q_idx < block_size * block_size; q_idx += WARP_SIZE) {
                const int q_h = q_idx / block_size;
                const int q_w = q_idx % block_size;
                const SpatialCoords q_pos = q_block.to_spatial(q_h, q_w);
                
                if (!q_pos.valid()) continue;
                
                for (int k_idx = 0; k_idx < block_size * block_size; k_idx++) {
                    const int k_h = k_idx / block_size;
                    const int k_w = k_idx % block_size;
                    const SpatialCoords k_pos = k_block.to_spatial(k_h, k_w);
                    
                    if (!k_pos.valid()) continue;
                    
                    // Compute attention score with SIMD
                    scalar_t score = 0;
                    #pragma unroll
                    for (int d = 0; d < head_dim; d += 4) {
                        score += __fmaf_rn(
                            shared_q[q_idx * head_dim + d],
                            shared_k[k_idx * head_dim + d],
                            0.f
                        );
                    }
                    score *= scale;
                    
                    // Add relative position bias
                    if (rel_pos_h && rel_pos_w) {
                        int rel_h, rel_w;
                        compute_2d_rel_pos_indices(
                            &rel_h, &rel_w,
                            q_pos, k_pos,
                            block_size
                        );
                        score += rel_pos_h[rel_h] + rel_pos_w[rel_w];
                    }
                    
                    // Store score
                    const int q_linear = q_pos.to_linear_idx();
                    const int k_linear = k_pos.to_linear_idx();
                    atomicAdd(
                        &scores[
                            ((batch_idx * gridDim.y + head_idx) * seq_length + q_linear) * 
                            seq_length + k_linear
                        ],
                        score
                    );
                }
            }
            
            __syncthreads();
        }
    } else {
        // Original 1D attention implementation
        const int q_block_start = block_idx * block_size;
        const int batch_offset = (batch_idx * gridDim.y + head_idx) * seq_length;
        
        if (q_block_start + thread_idx < seq_length) {
            load_to_shared_memory_vectorized(
                shared_q + thread_idx * head_dim,
                query + (batch_offset + q_block_start + thread_idx) * head_dim,
                0, 1, head_dim
            );
        }
        
        __syncthreads();
        
        for (int k_block_idx = warp_idx; k_block_idx < gridDim.z; k_block_idx += blockDim.x / WARP_SIZE) {
            const int selected_block = block_indices ? 
                block_indices[(batch_idx * gridDim.y + head_idx) * gridDim.z + k_block_idx] :
                k_block_idx;
            const int k_block_start = selected_block * block_size;
            
            if (k_block_start + lane_idx < seq_length) {
                load_to_shared_memory_vectorized(
                    shared_k + lane_idx * head_dim,
                    key + (batch_offset + k_block_start + lane_idx) * head_dim,
                    0, 1, head_dim
                );
            }
            
            __syncwarp();
            
            if (q_block_start + thread_idx < seq_length) {
                scalar_t q_vec[4];
                #pragma unroll
                for (int d = 0; d < head_dim; d += 4) {
                    *reinterpret_cast<float4*>(q_vec) = *reinterpret_cast<const float4*>(
                        shared_q + thread_idx * head_dim + d
                    );
                    
                    for (int k_offset = 0; k_offset < block_size && k_block_start + k_offset < seq_length; k_offset++) {
                        scalar_t k_vec[4];
                        *reinterpret_cast<float4*>(k_vec) = *reinterpret_cast<const float4*>(
                            shared_k + k_offset * head_dim + d
                        );
                        
                        scalar_t dot = 0;
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            dot += q_vec[i] * k_vec[i];
                        }
                        
                        const int score_idx = (
                            batch_offset + q_block_start + thread_idx
                        ) * seq_length + k_block_start + k_offset;
                        atomicAdd(&scores[score_idx], dot * scale);
                    }
                }
            }
            
            __syncthreads();
        }
    }
} 