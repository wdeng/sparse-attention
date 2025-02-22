#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <nvToolsExt.h>

// Constants for kernel optimization
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 256;
constexpr int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB shared memory per block
constexpr int L2_CACHE_LINE_SIZE = 128;       // L2 cache line size for coalescing
constexpr int HBM_BANK_SIZE = 256;            // HBM bank size for bank conflict avoidance

// 2D spatial attention constants
constexpr int MAX_SPATIAL_WINDOW = 49;  // 7x7 window
constexpr int MAX_SPATIAL_BLOCKS = 196;  // For 224x224 image with 16x16 blocks
constexpr int MIN_BLOCKS_PER_SM = 2;    // Minimum blocks per SM for occupancy

// Tensor Core optimization constants
constexpr int BLOCK_SIZE_M = 64;  // Tensor Core friendly sizes
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 8;
constexpr int MEMORY_ALIGNMENT = 128;  // 128-byte alignment for HBM efficiency

// Mixed precision type handling
template<typename T>
struct TypeInfo {
    static constexpr bool is_fp16 = false;
    static constexpr bool is_bf16 = false;
    static constexpr bool is_fp32 = false;
};

template<>
struct TypeInfo<half> {
    static constexpr bool is_fp16 = true;
};

template<>
struct TypeInfo<nv_bfloat16> {
    static constexpr bool is_bf16 = true;
};

template<>
struct TypeInfo<float> {
    static constexpr bool is_fp32 = true;
};

// Profiling utilities
struct ScopedNvtxRange {
    ScopedNvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~ScopedNvtxRange() {
        nvtxRangePop();
    }
};

#define PROFILE_KERNEL(name) ScopedNvtxRange _nvtx_range(name)

// Memory access patterns for HBM optimization
template<typename T>
struct alignas(MEMORY_ALIGNMENT) HBMAlignedBuffer {
    T data[HBM_BANK_SIZE / sizeof(T)];
};

// Helper functions for grid/block dimensions
inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

// Compute optimal grid dimensions for sparse attention with tensor core alignment
inline dim3 get_grid_dim(int batch_size, int num_heads, int seq_length, int block_size) {
    const int blocks_per_sequence = ceil_div(seq_length, BLOCK_SIZE_M);
    return dim3(batch_size, num_heads, blocks_per_sequence);
}

// Compute optimal block dimensions with warp and tensor core alignment
inline dim3 get_block_dim(int block_size) {
    // Round up to nearest warp size and ensure tensor core compatibility
    const int warps_per_block = ceil_div(block_size, WARP_SIZE);
    return dim3(min(warps_per_block * WARP_SIZE, MAX_THREADS_PER_BLOCK));
}

// Shared memory utilities with mixed precision support
template<typename T>
struct SharedMemory {
    __device__ inline T* get() {
        extern __shared__ unsigned char memory[] __align__(MEMORY_ALIGNMENT);
        if constexpr (TypeInfo<T>::is_fp16 || TypeInfo<T>::is_bf16) {
            // Add extra padding for half precision types
            return reinterpret_cast<T*>(memory) + 
                   (threadIdx.x % (HBM_BANK_SIZE / sizeof(T))) + 
                   (HBM_BANK_SIZE / sizeof(T));  // Extra padding for bank conflicts
        } else {
            return reinterpret_cast<T*>(memory) + 
                   threadIdx.x % (HBM_BANK_SIZE / sizeof(T));
        }
    }
};

// Optimized memory loading with vectorized access and mixed precision support
template<typename scalar_t>
__device__ inline void load_to_shared_memory_vectorized(
    scalar_t* shared_dest,
    const scalar_t* global_src,
    int offset,
    int stride,
    int size
) {
    if constexpr (TypeInfo<scalar_t>::is_fp16 || TypeInfo<scalar_t>::is_bf16) {
        // Use vectorized loads for half precision
        using Vec2 = typename std::aligned_storage<sizeof(scalar_t[2]), alignof(scalar_t[2])>::type;
        const Vec2* src_vec2 = reinterpret_cast<const Vec2*>(
            reinterpret_cast<const char*>(global_src + offset) + 
            (threadIdx.x * MEMORY_ALIGNMENT) % (HBM_BANK_SIZE * sizeof(scalar_t))
        );
        Vec2* dest_vec2 = reinterpret_cast<Vec2*>(
            reinterpret_cast<char*>(shared_dest) + 
            (threadIdx.x * MEMORY_ALIGNMENT) % (HBM_BANK_SIZE * sizeof(scalar_t))
        );
        
        #pragma unroll
        for (int i = 0; i < size / 2; i++) {
            dest_vec2[i] = src_vec2[i * stride / 2];
        }
        
        // Handle remaining elements
        if (size % 2) {
            shared_dest[size - 1] = global_src[offset + (size - 1) * stride];
        }
    } else {
        // Use float4 for full precision
        using Vec4 = typename std::aligned_storage<sizeof(scalar_t[4]), alignof(scalar_t[4])>::type;
        const Vec4* src_vec4 = reinterpret_cast<const Vec4*>(
            reinterpret_cast<const char*>(global_src + offset) + 
            (threadIdx.x * MEMORY_ALIGNMENT) % (HBM_BANK_SIZE * sizeof(scalar_t))
        );
        Vec4* dest_vec4 = reinterpret_cast<Vec4*>(
            reinterpret_cast<char*>(shared_dest) + 
            (threadIdx.x * MEMORY_ALIGNMENT) % (HBM_BANK_SIZE * sizeof(scalar_t))
        );
        
        #pragma unroll
        for (int i = 0; i < size / 4; i++) {
            dest_vec4[i] = src_vec4[i * stride / 4];
        }
        
        // Handle remaining elements
        for (int i = (size / 4) * 4; i < size; i++) {
            shared_dest[i] = global_src[offset + i * stride];
        }
    }
}

// Warp-level reduction utilities optimized for tensor cores
template<typename scalar_t>
__device__ inline scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction with tensor core alignment
template<typename scalar_t>
__device__ inline void block_reduce_sum(
    scalar_t* shared_mem,
    scalar_t thread_sum,
    int tid
) {
    // First warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store to shared memory
    if (tid % WARP_SIZE == 0) {
        shared_mem[tid / WARP_SIZE] = thread_sum;
    }
    
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < WARP_SIZE) {
        thread_sum = (tid < blockDim.x / WARP_SIZE) ? shared_mem[tid] : 0;
        thread_sum = warp_reduce_sum(thread_sum);
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

// 2D spatial coordinates helper
struct SpatialCoords {
    int h, w, H, W;
    
    __device__ inline int to_linear_idx() const {
        return h * W + w;
    }
    
    __device__ inline bool in_window(const SpatialCoords& other, int window_size) const {
        int dh = abs(h - other.h);
        int dw = abs(w - other.w);
        return dh <= window_size/2 && dw <= window_size/2;
    }
    
    __device__ inline bool valid() const {
        return h >= 0 && h < H && w >= 0 && w < W;
    }
};

// 2D block helper
struct BlockCoords {
    int bh, bw, block_size;
    
    __device__ inline SpatialCoords to_spatial(int idx_h, int idx_w) const {
        return {
            bh * block_size + idx_h,
            bw * block_size + idx_w,
            block_size,
            block_size
        };
    }
};

// Helper functions for grid/block dimensions with 2D support
inline dim3 get_grid_dim_2d(
    int batch_size,
    int num_heads,
    int height,
    int width,
    int block_size
) {
    const int blocks_h = (height + block_size - 1) / block_size;
    const int blocks_w = (width + block_size - 1) / block_size;
    return dim3(batch_size, num_heads, blocks_h * blocks_w);
}

// Compute optimal block dimensions with warp and tensor core alignment
inline dim3 get_block_dim_2d(int block_size) {
    // For 2D, we want multiple warps to handle each spatial block
    const int warps_per_block = min(4, (block_size * block_size + WARP_SIZE - 1) / WARP_SIZE);
    return dim3(warps_per_block * WARP_SIZE);
}

// Shared memory utilities with mixed precision and 2D support
template<typename T>
struct SharedMemory2D {
    __device__ inline T* get(int window_size) {
        extern __shared__ unsigned char memory[] __align__(MEMORY_ALIGNMENT);
        // Add padding for both height and width dimensions
        const int padding = TypeInfo<T>::is_fp16 || TypeInfo<T>::is_bf16 ? 2 : 1;
        return reinterpret_cast<T*>(memory) + 
               (threadIdx.x % (HBM_BANK_SIZE / sizeof(T))) +
               (window_size * window_size * padding);
    }
};

// Optimized 2D memory loading with vectorized access
template<typename scalar_t>
__device__ inline void load_2d_block_to_shared(
    scalar_t* shared_dest,
    const scalar_t* global_src,
    const BlockCoords& coords,
    int height,
    int width,
    int stride
) {
    const int tid = threadIdx.x;
    const int block_size = coords.block_size;
    const int block_area = block_size * block_size;
    
    // Each thread loads multiple elements using vectorized access
    if constexpr (TypeInfo<scalar_t>::is_fp16 || TypeInfo<scalar_t>::is_bf16) {
        using Vec2 = typename std::aligned_storage<sizeof(scalar_t[2]), alignof(scalar_t[2])>::type;
        
        for (int idx = tid * 2; idx < block_area; idx += blockDim.x * 2) {
            const int h = idx / block_size;
            const int w = idx % block_size;
            const int gh = coords.bh * block_size + h;
            const int gw = coords.bw * block_size + w;
            
            if (gh < height && gw < width) {
                const Vec2* src_vec2 = reinterpret_cast<const Vec2*>(
                    &global_src[(gh * width + gw) * stride]
                );
                Vec2* dst_vec2 = reinterpret_cast<Vec2*>(
                    &shared_dest[h * block_size + w]
                );
                *dst_vec2 = *src_vec2;
            }
        }
    } else {
        using Vec4 = typename std::aligned_storage<sizeof(scalar_t[4]), alignof(scalar_t[4])>::type;
        
        for (int idx = tid * 4; idx < block_area; idx += blockDim.x * 4) {
            const int h = idx / block_size;
            const int w = idx % block_size;
            const int gh = coords.bh * block_size + h;
            const int gw = coords.bw * block_size + w;
            
            if (gh < height && gw < width) {
                const Vec4* src_vec4 = reinterpret_cast<const Vec4*>(
                    &global_src[(gh * width + gw) * stride]
                );
                Vec4* dst_vec4 = reinterpret_cast<Vec4*>(
                    &shared_dest[h * block_size + w]
                );
                *dst_vec4 = *src_vec4;
            }
        }
    }
}

// Compute relative position encoding indices for 2D attention
__device__ inline void compute_2d_rel_pos_indices(
    int* rel_pos_h,
    int* rel_pos_w,
    const SpatialCoords& q_pos,
    const SpatialCoords& k_pos,
    int spatial_window
) {
    const int dh = k_pos.h - q_pos.h + spatial_window - 1;
    const int dw = k_pos.w - q_pos.w + spatial_window - 1;
    *rel_pos_h = min(max(dh, 0), 2 * spatial_window - 2);
    *rel_pos_w = min(max(dw, 0), 2 * spatial_window - 2);
} 