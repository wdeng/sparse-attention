#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Constants for kernel optimization
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 256;
constexpr int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB shared memory per block
constexpr int L2_CACHE_LINE_SIZE = 128;       // L2 cache line size for coalescing
constexpr int HBM_BANK_SIZE = 256;            // HBM bank size for bank conflict avoidance

// Memory access patterns for HBM optimization
template<typename T>
struct alignas(HBM_BANK_SIZE) HBMAlignedBuffer {
    T data[HBM_BANK_SIZE / sizeof(T)];
};

// Helper functions for grid/block dimensions
inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// Compute optimal grid dimensions for sparse attention
inline dim3 get_grid_dim(int batch_size, int num_heads, int seq_length, int block_size) {
    const int blocks_per_sequence = ceil_div(seq_length, block_size);
    return dim3(batch_size, num_heads, blocks_per_sequence);
}

// Compute optimal block dimensions with warp alignment
inline dim3 get_block_dim(int block_size) {
    // Round up to nearest warp size for coalesced memory access
    const int warps_per_block = ceil_div(block_size, WARP_SIZE);
    return dim3(min(warps_per_block * WARP_SIZE, MAX_THREADS_PER_BLOCK));
}

// Shared memory utilities with bank conflict avoidance
template<typename T>
struct SharedMemory {
    __device__ inline T* get() {
        extern __shared__ unsigned char memory[];
        // Add padding to avoid bank conflicts
        return reinterpret_cast<T*>(memory) + threadIdx.x % (HBM_BANK_SIZE / sizeof(T));
    }
};

// Optimized memory loading with vectorized access
template<typename scalar_t>
__device__ inline void load_to_shared_memory_vectorized(
    scalar_t* shared_dest,
    const scalar_t* global_src,
    int offset,
    int stride,
    int size
) {
    using Vec4 = typename std::aligned_storage<sizeof(scalar_t[4]), alignof(scalar_t[4])>::type;
    const Vec4* src_vec4 = reinterpret_cast<const Vec4*>(global_src + offset);
    Vec4* dest_vec4 = reinterpret_cast<Vec4*>(shared_dest);
    
    #pragma unroll
    for (int i = 0; i < size / 4; i++) {
        dest_vec4[i] = src_vec4[i * stride / 4];
    }
    
    // Handle remaining elements
    for (int i = (size / 4) * 4; i < size; i++) {
        shared_dest[i] = global_src[offset + i * stride];
    }
}

// CUDA kernel for computing block-level attention scores with HBM optimizations
template<typename scalar_t>
__global__ void compute_block_attention_scores_kernel(
    const scalar_t* __restrict__ query,      // [batch, heads, seq_len, head_dim]
    const scalar_t* __restrict__ key,        // [batch, heads, seq_len, head_dim]
    scalar_t* __restrict__ scores,           // [batch, heads, seq_len, seq_len]
    const int* __restrict__ block_indices,   // [batch, heads, num_blocks]
    const int seq_length,
    const int head_dim,
    const int block_size,
    const float scale
) {
    // Shared memory allocation with padding for bank conflict avoidance
    extern __shared__ char shared_memory[];
    scalar_t* shared_q = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_k = shared_q + blockDim.x * head_dim + HBM_BANK_SIZE / sizeof(scalar_t);
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_idx = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;
    
    // Compute base indices for coalesced memory access
    const int q_block_start = block_idx * block_size;
    const int batch_offset = (batch_idx * gridDim.y + head_idx) * seq_length;
    
    // Load query block into shared memory using vectorized loads
    if (q_block_start + thread_idx < seq_length) {
        load_to_shared_memory_vectorized(
            shared_q + thread_idx * head_dim,
            query + (batch_offset + q_block_start + thread_idx) * head_dim,
            0, 1, head_dim
        );
    }
    
    __syncthreads();
    
    // Process key blocks with memory coalescing
    for (int k_block_idx = warp_idx; k_block_idx < gridDim.z; k_block_idx += blockDim.x / WARP_SIZE) {
        const int selected_block = block_indices[
            (batch_idx * gridDim.y + head_idx) * gridDim.z + k_block_idx
        ];
        const int k_block_start = selected_block * block_size;
        
        // Load key block with vectorized access
        if (k_block_start + lane_idx < seq_length) {
            load_to_shared_memory_vectorized(
                shared_k + lane_idx * head_dim,
                key + (batch_offset + k_block_start + lane_idx) * head_dim,
                0, 1, head_dim
            );
        }
        
        __syncwarp();
        
        // Compute attention scores with SIMD instructions
        if (q_block_start + thread_idx < seq_length) {
            scalar_t q_vec[4];  // Load 4 elements at once
            #pragma unroll
            for (int d = 0; d < head_dim; d += 4) {
                // Load query vector
                *reinterpret_cast<float4*>(q_vec) = *reinterpret_cast<const float4*>(
                    shared_q + thread_idx * head_dim + d
                );
                
                for (int k_offset = 0; k_offset < block_size && k_block_start + k_offset < seq_length; k_offset++) {
                    scalar_t k_vec[4];
                    // Load key vector
                    *reinterpret_cast<float4*>(k_vec) = *reinterpret_cast<const float4*>(
                        shared_k + k_offset * head_dim + d
                    );
                    
                    // Compute dot product with SIMD
                    scalar_t dot = 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        dot += q_vec[i] * k_vec[i];
                    }
                    
                    // Accumulate score
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

// CUDA kernel for computing attention output with HBM optimizations
template<typename scalar_t>
__global__ void compute_attention_output_kernel(
    const scalar_t* __restrict__ attn_weights,  // [batch, heads, seq_len, seq_len]
    const scalar_t* __restrict__ value,         // [batch, heads, seq_len, head_dim]
    scalar_t* __restrict__ output,              // [batch, heads, seq_len, head_dim]
    const int* __restrict__ block_indices,      // [batch, heads, num_blocks]
    const int seq_length,
    const int head_dim,
    const int block_size
) {
    // Shared memory with padding for bank conflict avoidance
    extern __shared__ char shared_memory[];
    scalar_t* shared_weights = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* shared_values = shared_weights + blockDim.x * block_size + HBM_BANK_SIZE / sizeof(scalar_t);
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_idx = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;
    
    const int q_idx = block_idx * block_size + thread_idx;
    const int batch_offset = (batch_idx * gridDim.y + head_idx) * seq_length;
    
    // Initialize output accumulator with vectorized access
    alignas(16) scalar_t acc[64];  // Aligned for vectorized operations
    #pragma unroll
    for (int d = 0; d < head_dim; d += 4) {
        *reinterpret_cast<float4*>(&acc[d]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    
    // Process blocks with warp-level parallelism
    for (int k_block_idx = warp_idx; k_block_idx < gridDim.z; k_block_idx += blockDim.x / WARP_SIZE) {
        const int selected_block = block_indices[
            (batch_idx * gridDim.y + head_idx) * gridDim.z + k_block_idx
        ];
        const int k_block_start = selected_block * block_size;
        
        // Load attention weights with vectorized access
        if (q_idx < seq_length) {
            load_to_shared_memory_vectorized(
                shared_weights + thread_idx * block_size,
                attn_weights + (batch_offset + q_idx) * seq_length + k_block_start,
                0, 1, block_size
            );
        }
        
        // Load values with vectorized access
        if (lane_idx < block_size && k_block_start + lane_idx < seq_length) {
            load_to_shared_memory_vectorized(
                shared_values + lane_idx * head_dim,
                value + (batch_offset + k_block_start + lane_idx) * head_dim,
                0, 1, head_dim
            );
        }
        
        __syncwarp();
        
        // Compute weighted sum with SIMD instructions
        if (q_idx < seq_length) {
            #pragma unroll
            for (int k_offset = 0; k_offset < block_size && k_block_start + k_offset < seq_length; k_offset++) {
                const scalar_t weight = shared_weights[thread_idx * block_size + k_offset];
                
                // Vectorized accumulation
                for (int d = 0; d < head_dim; d += 4) {
                    float4 value_vec = *reinterpret_cast<const float4*>(
                        shared_values + k_offset * head_dim + d
                    );
                    float4 acc_vec = *reinterpret_cast<float4*>(&acc[d]);
                    
                    acc_vec.x += weight * value_vec.x;
                    acc_vec.y += weight * value_vec.y;
                    acc_vec.z += weight * value_vec.z;
                    acc_vec.w += weight * value_vec.w;
                    
                    *reinterpret_cast<float4*>(&acc[d]) = acc_vec;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store final output with vectorized writes
    if (q_idx < seq_length) {
        const int out_base = (batch_offset + q_idx) * head_dim;
        #pragma unroll
        for (int d = 0; d < head_dim; d += 4) {
            *reinterpret_cast<float4*>(&output[out_base + d]) = 
                *reinterpret_cast<float4*>(&acc[d]);
        }
    }
} 