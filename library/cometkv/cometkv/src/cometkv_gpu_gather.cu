#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cometkv_gpu_gather_kernel.cuh"

// Host wrappers for the CometKV_GPU backend: gather a [static | recent | sparse]
// FlashAttention buffer where the sparse retrieval store lives entirely on GPU.
// Isolated from the UVA gather (cometkv_gather.cu) so the device-resident path
// can be tuned without touching the production host-store kernel.

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_two_byte_dtype(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.element_size() == 2, name, " must use a 2-byte dtype");
}

}  // namespace


void concat_static_recent_gpu_gather(
    torch::Tensor static_keys,
    torch::Tensor static_values,
    torch::Tensor recent_keys,
    torch::Tensor recent_values,
    torch::Tensor request_token_ids,
    torch::Tensor gpu_kv,
    torch::Tensor out_keys,
    torch::Tensor out_values,
    int64_t static_len,
    int64_t recent_len,
    int64_t sparse_len) {
    check_cuda_tensor(static_keys, "static_keys");
    check_cuda_tensor(static_values, "static_values");
    check_cuda_tensor(recent_keys, "recent_keys");
    check_cuda_tensor(recent_values, "recent_values");
    check_cuda_tensor(request_token_ids, "request_token_ids");
    check_cuda_tensor(gpu_kv, "gpu_kv");
    check_cuda_tensor(out_keys, "out_keys");
    check_cuda_tensor(out_values, "out_values");
    check_two_byte_dtype(static_keys, "static_keys");
    check_two_byte_dtype(static_values, "static_values");
    check_two_byte_dtype(recent_keys, "recent_keys");
    check_two_byte_dtype(recent_values, "recent_values");
    check_two_byte_dtype(gpu_kv, "gpu_kv");
    check_two_byte_dtype(out_keys, "out_keys");
    check_two_byte_dtype(out_values, "out_values");

    TORCH_CHECK(request_token_ids.scalar_type() == torch::kInt32, "request_token_ids must be int32");
    TORCH_CHECK(static_keys.scalar_type() == static_values.scalar_type(), "static key/value dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == recent_keys.scalar_type(), "recent key dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == recent_values.scalar_type(), "recent value dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == out_keys.scalar_type(), "output key dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == out_values.scalar_type(), "output value dtype mismatch");
    TORCH_CHECK(static_keys.dim() == 3 && static_values.dim() == 3, "static tensors must be [rows, len, dim]");
    TORCH_CHECK(recent_keys.dim() == 3 && recent_values.dim() == 3, "recent tensors must be [rows, len, dim]");
    TORCH_CHECK(request_token_ids.dim() == 2, "request_token_ids must be [rows, request_count]");
    TORCH_CHECK(gpu_kv.dim() == 4, "gpu_kv must be [rows, total_tokens, 2, dim]");
    TORCH_CHECK(out_keys.dim() == 4 && out_values.dim() == 4, "output tensors must be [rows, len, 1, dim]");

    const int rows = static_cast<int>(static_keys.size(0));
    const int static_capacity = static_cast<int>(static_keys.size(1));
    const int recent_capacity = static_cast<int>(recent_keys.size(1));
    const int dim = static_cast<int>(static_keys.size(2));
    const int request_count = static_cast<int>(request_token_ids.size(1));
    const int total_tokens = static_cast<int>(gpu_kv.size(1));
    const int total_len = static_cast<int>(static_len + recent_len + sparse_len);

    TORCH_CHECK(dim > 0, "dim must be positive");
    TORCH_CHECK(static_len >= 0 && recent_len >= 0 && sparse_len >= 0, "concat lengths must be non-negative");
    TORCH_CHECK(static_len <= static_capacity, "static_len exceeds static capacity");
    TORCH_CHECK(recent_len <= recent_capacity, "recent_len exceeds recent capacity");
    TORCH_CHECK(sparse_len <= request_count, "sparse_len exceeds request_count");
    TORCH_CHECK(static_values.size(0) == rows && static_values.size(1) == static_capacity && static_values.size(2) == dim,
                "static_values shape mismatch");
    TORCH_CHECK(recent_keys.size(0) == rows && recent_values.size(0) == rows, "recent row mismatch");
    TORCH_CHECK(recent_values.size(1) == recent_capacity && recent_keys.size(2) == dim && recent_values.size(2) == dim,
                "recent shape mismatch");
    TORCH_CHECK(request_token_ids.size(0) == rows, "request row mismatch");
    TORCH_CHECK(gpu_kv.size(0) == rows && gpu_kv.size(2) == 2 && gpu_kv.size(3) == dim, "gpu_kv shape mismatch");
    TORCH_CHECK(out_keys.size(0) == rows && out_values.size(0) == rows, "output row mismatch");
    TORCH_CHECK(out_keys.size(1) >= total_len && out_values.size(1) == out_keys.size(1),
                "output rows must be at least total_len wide (out_row_len >= total_len)");
    const int out_row_len = static_cast<int>(out_keys.size(1));
    TORCH_CHECK(out_keys.size(2) == 1 && out_values.size(2) == 1, "output head dimension must be 1");
    TORCH_CHECK(out_keys.size(3) == dim && out_values.size(3) == dim, "output dim mismatch");

    if (total_len == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(rows, total_len);
    cometkv_gpu::concat_static_recent_gpu_gather_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(static_keys.data_ptr()),
        reinterpret_cast<const uint16_t*>(static_values.data_ptr()),
        reinterpret_cast<const uint16_t*>(recent_keys.data_ptr()),
        reinterpret_cast<const uint16_t*>(recent_values.data_ptr()),
        request_token_ids.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(gpu_kv.data_ptr()),
        reinterpret_cast<uint16_t*>(out_keys.data_ptr()),
        reinterpret_cast<uint16_t*>(out_values.data_ptr()),
        rows,
        static_capacity,
        recent_capacity,
        request_count,
        total_tokens,
        static_cast<int>(static_len),
        static_cast<int>(recent_len),
        static_cast<int>(sparse_len),
        total_len,
        out_row_len,
        dim
    );
}


void concat_static_recent_gpu_gather_int8(
    torch::Tensor static_keys,
    torch::Tensor static_values,
    torch::Tensor recent_keys,
    torch::Tensor recent_values,
    torch::Tensor request_token_ids,
    torch::Tensor gpu_kv_int8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor out_keys,
    torch::Tensor out_values,
    int64_t static_len,
    int64_t recent_len,
    int64_t sparse_len) {
    check_cuda_tensor(static_keys, "static_keys");
    check_cuda_tensor(static_values, "static_values");
    check_cuda_tensor(recent_keys, "recent_keys");
    check_cuda_tensor(recent_values, "recent_values");
    check_cuda_tensor(request_token_ids, "request_token_ids");
    check_cuda_tensor(gpu_kv_int8, "gpu_kv_int8");
    check_cuda_tensor(k_scale, "k_scale");
    check_cuda_tensor(v_scale, "v_scale");
    check_cuda_tensor(out_keys, "out_keys");
    check_cuda_tensor(out_values, "out_values");
    check_two_byte_dtype(static_keys, "static_keys");
    check_two_byte_dtype(static_values, "static_values");
    check_two_byte_dtype(recent_keys, "recent_keys");
    check_two_byte_dtype(recent_values, "recent_values");
    check_two_byte_dtype(out_keys, "out_keys");
    check_two_byte_dtype(out_values, "out_values");

    TORCH_CHECK(gpu_kv_int8.scalar_type() == torch::kChar, "gpu_kv_int8 must be int8");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32, "k_scale must be float32");
    TORCH_CHECK(v_scale.scalar_type() == torch::kFloat32, "v_scale must be float32");
    TORCH_CHECK(request_token_ids.scalar_type() == torch::kInt32, "request_token_ids must be int32");
    TORCH_CHECK(static_keys.scalar_type() == out_keys.scalar_type(), "static/out key dtype mismatch");
    TORCH_CHECK(static_keys.dim() == 3 && static_values.dim() == 3, "static tensors must be [rows, len, dim]");
    TORCH_CHECK(recent_keys.dim() == 3 && recent_values.dim() == 3, "recent tensors must be [rows, len, dim]");
    TORCH_CHECK(request_token_ids.dim() == 2, "request_token_ids must be [rows, request_count]");
    TORCH_CHECK(gpu_kv_int8.dim() == 4, "gpu_kv_int8 must be [rows, total_tokens, 2, dim]");
    TORCH_CHECK(out_keys.dim() == 4 && out_values.dim() == 4, "output tensors must be [rows, len, 1, dim]");

    const int rows = static_cast<int>(static_keys.size(0));
    const int static_capacity = static_cast<int>(static_keys.size(1));
    const int recent_capacity = static_cast<int>(recent_keys.size(1));
    const int dim = static_cast<int>(static_keys.size(2));
    const int request_count = static_cast<int>(request_token_ids.size(1));
    const int total_tokens = static_cast<int>(gpu_kv_int8.size(1));
    const int total_len = static_cast<int>(static_len + recent_len + sparse_len);

    TORCH_CHECK(dim > 0, "dim must be positive");
    TORCH_CHECK(static_len >= 0 && recent_len >= 0 && sparse_len >= 0, "concat lengths must be non-negative");
    TORCH_CHECK(static_len <= static_capacity, "static_len exceeds static capacity");
    TORCH_CHECK(recent_len <= recent_capacity, "recent_len exceeds recent capacity");
    TORCH_CHECK(sparse_len <= request_count, "sparse_len exceeds request_count");
    TORCH_CHECK(gpu_kv_int8.size(0) == rows && gpu_kv_int8.size(2) == 2 && gpu_kv_int8.size(3) == dim,
                "gpu_kv_int8 shape mismatch");
    TORCH_CHECK(k_scale.dim() == 2 && k_scale.size(0) == rows && k_scale.size(1) == dim,
                "k_scale must be [rows, dim]");
    TORCH_CHECK(v_scale.dim() == 2 && v_scale.size(0) == rows && v_scale.size(1) == total_tokens,
                "v_scale must be [rows, total_tokens]");
    TORCH_CHECK(out_keys.size(0) == rows && out_keys.size(1) >= total_len && out_keys.size(2) == 1 &&
                out_keys.size(3) == dim, "out_keys shape mismatch");
    TORCH_CHECK(out_values.size(1) == out_keys.size(1), "out_values width mismatch");
    const int out_row_len = static_cast<int>(out_keys.size(1));

    if (total_len == 0) {
        return;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(rows, total_len);
    if (out_keys.scalar_type() == torch::kFloat16) {
        cometkv_gpu::concat_static_recent_gpu_gather_int8_kernel<half><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const half*>(static_keys.data_ptr()),
            reinterpret_cast<const half*>(static_values.data_ptr()),
            reinterpret_cast<const half*>(recent_keys.data_ptr()),
            reinterpret_cast<const half*>(recent_values.data_ptr()),
            request_token_ids.data_ptr<int32_t>(),
            gpu_kv_int8.data_ptr<int8_t>(),
            k_scale.data_ptr<float>(),
            v_scale.data_ptr<float>(),
            reinterpret_cast<half*>(out_keys.data_ptr()),
            reinterpret_cast<half*>(out_values.data_ptr()),
            rows, static_capacity, recent_capacity, request_count, total_tokens,
            static_cast<int>(static_len), static_cast<int>(recent_len), static_cast<int>(sparse_len),
            total_len, out_row_len, dim);
    } else if (out_keys.scalar_type() == torch::kBFloat16) {
        cometkv_gpu::concat_static_recent_gpu_gather_int8_kernel<nv_bfloat16><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(static_keys.data_ptr()),
            reinterpret_cast<const nv_bfloat16*>(static_values.data_ptr()),
            reinterpret_cast<const nv_bfloat16*>(recent_keys.data_ptr()),
            reinterpret_cast<const nv_bfloat16*>(recent_values.data_ptr()),
            request_token_ids.data_ptr<int32_t>(),
            gpu_kv_int8.data_ptr<int8_t>(),
            k_scale.data_ptr<float>(),
            v_scale.data_ptr<float>(),
            reinterpret_cast<nv_bfloat16*>(out_keys.data_ptr()),
            reinterpret_cast<nv_bfloat16*>(out_values.data_ptr()),
            rows, static_capacity, recent_capacity, request_count, total_tokens,
            static_cast<int>(static_len), static_cast<int>(recent_len), static_cast<int>(sparse_len),
            total_len, out_row_len, dim);
    } else {
        TORCH_CHECK(false, "out_keys must be fp16 or bf16");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("concat_static_recent_gpu_gather", &concat_static_recent_gpu_gather,
          "Concat static/recent KV and gather sparse KV from a GPU-resident store into a FlashAttention buffer (CUDA)",
          py::arg("static_keys"), py::arg("static_values"), py::arg("recent_keys"), py::arg("recent_values"),
          py::arg("request_token_ids"), py::arg("gpu_kv"), py::arg("out_keys"), py::arg("out_values"),
          py::arg("static_len"), py::arg("recent_len"), py::arg("sparse_len"));
    m.def("concat_static_recent_gpu_gather_int8", &concat_static_recent_gpu_gather_int8,
          "int8 GPU-resident gather: dequant K per-channel / V per-token to bf16/fp16 (CUDA)",
          py::arg("static_keys"), py::arg("static_values"), py::arg("recent_keys"), py::arg("recent_values"),
          py::arg("request_token_ids"), py::arg("gpu_kv_int8"), py::arg("k_scale"), py::arg("v_scale"),
          py::arg("out_keys"), py::arg("out_values"),
          py::arg("static_len"), py::arg("recent_len"), py::arg("sparse_len"));
}
