#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// Forward declarations of CUDA functions
torch::Tensor sparse_attention_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const float scale,
    const int block_size
);

torch::Tensor sparse_attention_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const torch::Tensor& attn_weights,
    const float scale,
    const int block_size
);

// PyBind11 wrapper functions
torch::Tensor sparse_attention_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const float scale,
    const int block_size
) {
    // Input validation
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(key.is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(value.is_cuda(), "Value tensor must be on CUDA device");
    TORCH_CHECK(block_indices.is_cuda(), "Block indices tensor must be on CUDA device");
    
    TORCH_CHECK(query.dim() == 4, "Query tensor must be 4-dimensional");
    TORCH_CHECK(key.dim() == 4, "Key tensor must be 4-dimensional");
    TORCH_CHECK(value.dim() == 4, "Value tensor must be 4-dimensional");
    TORCH_CHECK(block_indices.dim() == 3, "Block indices tensor must be 3-dimensional");
    
    return sparse_attention_forward_cuda(
        query, key, value, block_indices, scale, block_size
    );
}

torch::Tensor sparse_attention_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_indices,
    const torch::Tensor& attn_weights,
    const float scale,
    const int block_size
) {
    // Input validation
    TORCH_CHECK(grad_output.is_cuda(), "Gradient output tensor must be on CUDA device");
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(key.is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(value.is_cuda(), "Value tensor must be on CUDA device");
    TORCH_CHECK(block_indices.is_cuda(), "Block indices tensor must be on CUDA device");
    TORCH_CHECK(attn_weights.is_cuda(), "Attention weights tensor must be on CUDA device");
    
    return sparse_attention_backward_cuda(
        grad_output, query, key, value, block_indices, attn_weights,
        scale, block_size
    );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention_forward", &sparse_attention_forward,
          "Forward pass for sparse attention with block selection");
    m.def("sparse_attention_backward", &sparse_attention_backward,
          "Backward pass for sparse attention with block selection");
} 