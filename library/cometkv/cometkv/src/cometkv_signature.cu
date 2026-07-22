#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <vector>

#include "cometkv_signature_kernel.cuh"

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_tensor_allow_strided(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_int32_tensor(const torch::Tensor& tensor, const char* name) {
    check_cuda_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
}

}  // namespace


void asym_signature_score_into(
    torch::Tensor query_proj,
    torch::Tensor query_proj_total,
    torch::Tensor keys,
    torch::Tensor range_starts,
    torch::Tensor range_ends,
    torch::Tensor scores,
    int64_t sig_bits_used,
    int64_t active_candidates,
    c10::optional<torch::Tensor> norm_lo,
    c10::optional<torch::Tensor> norm_step) {
    check_cuda_tensor(query_proj, "query_proj");
    check_cuda_tensor(query_proj_total, "query_proj_total");
    check_cuda_tensor(keys, "keys");
    check_int32_tensor(range_starts, "range_starts");
    check_int32_tensor(range_ends, "range_ends");
    check_cuda_tensor(scores, "scores");
    TORCH_CHECK(query_proj.scalar_type() == torch::kFloat32, "query_proj must be fp32");
    TORCH_CHECK(query_proj_total.scalar_type() == torch::kFloat32, "query_proj_total must be fp32");
    TORCH_CHECK(keys.scalar_type() == torch::kUInt8, "keys must be uint8");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be fp32");
    TORCH_CHECK(keys.dim() == 3, "keys must be [rows, tokens, sig_bytes]");

    const int64_t rows = keys.size(0);
    const int64_t tokens = keys.size(1);
    const int64_t sig_bytes = keys.size(2);
    const int64_t proj_width = query_proj.size(1);
    TORCH_CHECK(query_proj.sizes().vec() == std::vector<int64_t>({rows, proj_width}), "query_proj must be [rows, width]");
    TORCH_CHECK(query_proj_total.sizes().vec() == std::vector<int64_t>({rows}), "query_proj_total must be [rows]");
    TORCH_CHECK(scores.sizes().vec() == std::vector<int64_t>({rows, tokens}), "scores must be [rows, tokens]");
    TORCH_CHECK(range_starts.sizes().vec() == std::vector<int64_t>({rows, cometkv::kMaxRanges}), "range_starts must be [rows, 2]");
    TORCH_CHECK(range_ends.sizes().vec() == std::vector<int64_t>({rows, cometkv::kMaxRanges}), "range_ends must be [rows, 2]");
    TORCH_CHECK(sig_bits_used > 0 && sig_bits_used <= cometkv::kMaxHammingDistance, "sig_bits_used must be in (0, 128]");
    TORCH_CHECK(sig_bits_used % 8 == 0, "sig_bits_used must be a multiple of 8");
    TORCH_CHECK(sig_bits_used <= proj_width * 1, "sig_bits_used must fit query_proj width");
    TORCH_CHECK(sig_bits_used / 8 <= sig_bytes, "sig_bits_used must fit signature bytes");

    const float* norm_lo_ptr = nullptr;
    const float* norm_step_ptr = nullptr;
    if (norm_lo.has_value() || norm_step.has_value()) {
        TORCH_CHECK(norm_lo.has_value() && norm_step.has_value(), "norm_lo/norm_step must be passed together");
        check_cuda_tensor(norm_lo.value(), "norm_lo");
        check_cuda_tensor(norm_step.value(), "norm_step");
        TORCH_CHECK(norm_lo->scalar_type() == torch::kFloat32 && norm_step->scalar_type() == torch::kFloat32,
                    "norm_lo/norm_step must be fp32");
        TORCH_CHECK(norm_lo->sizes().vec() == std::vector<int64_t>({rows}), "norm_lo must be [rows]");
        TORCH_CHECK(norm_step->sizes().vec() == std::vector<int64_t>({rows}), "norm_step must be [rows]");
        TORCH_CHECK(sig_bits_used / 8 < sig_bytes, "asym_n8 needs a spare signature byte for the norm code");
        norm_lo_ptr = norm_lo->data_ptr<float>();
        norm_step_ptr = norm_step->data_ptr<float>();
    }

    int64_t candidates = active_candidates >= 0 ? std::min<int64_t>(active_candidates, tokens) : tokens;
    const int64_t max_chunks = (candidates + cometkv::kHammingTopkChunkSize - 1) / cometkv::kHammingTopkChunkSize;
    if (rows == 0 || max_chunks == 0) {
        return;
    }
    dim3 grid(rows, max_chunks);
    auto stream = at::cuda::getCurrentCUDAStream();
    cometkv::asym_signature_score_kernel<<<grid, cometkv::kHammingTopkThreads, 0, stream>>>(
        query_proj.data_ptr<float>(),
        query_proj_total.data_ptr<float>(),
        keys.data_ptr<uint8_t>(),
        range_starts.data_ptr<int32_t>(),
        range_ends.data_ptr<int32_t>(),
        norm_lo_ptr,
        norm_step_ptr,
        scores.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(tokens),
        static_cast<int>(sig_bytes),
        static_cast<int>(sig_bits_used),
        static_cast<int>(proj_width),
        static_cast<int>(max_chunks));
}


void sampled_tail_attention_merge(
    torch::Tensor queries,      // [rows, group, dim] bf16
    torch::Tensor tail_keys,    // [rows, m, dim] bf16
    torch::Tensor tail_values,  // [rows, m, dim] bf16
    torch::Tensor corr,         // [rows, m] fp32
    torch::Tensor out_main,     // [rows, group, dim] bf16 (in/out)
    torch::Tensor lse_main,     // [rows, group] fp32
    double scale,
    double clip) {
    check_cuda_tensor(queries, "queries");
    check_cuda_tensor(tail_keys, "tail_keys");
    check_cuda_tensor(tail_values, "tail_values");
    check_cuda_tensor(corr, "corr");
    check_cuda_tensor(out_main, "out_main");
    check_cuda_tensor(lse_main, "lse_main");
    TORCH_CHECK(queries.scalar_type() == torch::kBFloat16, "queries must be bf16");
    TORCH_CHECK(tail_keys.scalar_type() == torch::kBFloat16, "tail_keys must be bf16");
    TORCH_CHECK(tail_values.scalar_type() == torch::kBFloat16, "tail_values must be bf16");
    TORCH_CHECK(out_main.scalar_type() == torch::kBFloat16, "out_main must be bf16");
    TORCH_CHECK(corr.scalar_type() == torch::kFloat32, "corr must be fp32");
    TORCH_CHECK(lse_main.scalar_type() == torch::kFloat32, "lse_main must be fp32");
    TORCH_CHECK(queries.dim() == 3 && tail_keys.dim() == 3 && tail_values.dim() == 3,
                "queries/tail_keys/tail_values must be 3D");
    TORCH_CHECK(queries.is_contiguous() && tail_keys.is_contiguous()
                && tail_values.is_contiguous() && corr.is_contiguous()
                && out_main.is_contiguous() && lse_main.is_contiguous(),
                "all sampled-tail tensors must be contiguous");
    const int64_t rows = queries.size(0);
    const int64_t group = queries.size(1);
    const int64_t dim = queries.size(2);
    const int64_t m = tail_keys.size(1);
    TORCH_CHECK(tail_keys.size(0) == rows && tail_values.size(0) == rows, "rows mismatch");
    TORCH_CHECK(tail_keys.size(2) == dim && tail_values.size(2) == dim, "dim mismatch");
    TORCH_CHECK(tail_values.size(1) == m, "tail_values m mismatch");
    TORCH_CHECK(corr.size(0) == rows && corr.size(1) == m, "corr must be [rows, m]");
    TORCH_CHECK(out_main.sizes() == queries.sizes(), "out_main must match queries shape");
    TORCH_CHECK(lse_main.size(0) == rows && lse_main.size(1) == group, "lse_main must be [rows, group]");
    TORCH_CHECK(dim <= 256, "head_dim must be <= 256");
    if (rows == 0 || m == 0) {
        return;
    }
    const size_t smem = static_cast<size_t>(dim + m) * sizeof(float);
    TORCH_CHECK(smem <= 48 * 1024, "sampled tail m too large for shared memory");
    dim3 grid(rows, group);
    auto stream = at::cuda::getCurrentCUDAStream();
    cometkv::sampled_tail_attention_merge_kernel<<<grid, 128, smem, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(queries.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(tail_keys.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(tail_values.data_ptr()),
        corr.data_ptr<float>(),
        reinterpret_cast<nv_bfloat16*>(out_main.data_ptr()),
        lse_main.data_ptr<float>(),
        static_cast<float>(scale),
        static_cast<float>(clip),
        static_cast<int>(group),
        static_cast<int>(m),
        static_cast<int>(dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void uva_gather_kv_rows(
    torch::Tensor indices,   // [rows, m_cap] int32 (device)
    torch::Tensor cpu_kv,    // [rows, tokens, 2, dim] bf16 pinned host
    torch::Tensor out_k,     // [rows, m_cap, dim] bf16 (device)
    torch::Tensor out_v,     // [rows, m_cap, dim] bf16 (device)
    int64_t m) {
    check_int32_tensor(indices, "indices");
    check_cuda_tensor(out_k, "out_k");
    check_cuda_tensor(out_v, "out_v");
    TORCH_CHECK(cpu_kv.is_pinned() || cpu_kv.is_cuda(), "cpu_kv must be pinned host or device memory");
    TORCH_CHECK(cpu_kv.scalar_type() == torch::kBFloat16, "cpu_kv must be bf16");
    TORCH_CHECK(out_k.scalar_type() == torch::kBFloat16 && out_v.scalar_type() == torch::kBFloat16,
                "out_k/out_v must be bf16");
    TORCH_CHECK(cpu_kv.dim() == 4 && cpu_kv.size(2) == 2, "cpu_kv must be [rows, tokens, 2, dim]");
    TORCH_CHECK(indices.dim() == 2 && out_k.dim() == 3 && out_v.dim() == 3, "shape ranks mismatch");
    TORCH_CHECK(indices.is_contiguous() && out_k.is_contiguous() && out_v.is_contiguous()
                && cpu_kv.is_contiguous(), "all tensors must be contiguous");
    const int64_t rows = indices.size(0);
    const int64_t dim = cpu_kv.size(3);
    TORCH_CHECK(cpu_kv.size(0) == rows && out_k.size(0) == rows && out_v.size(0) == rows, "rows mismatch");
    TORCH_CHECK(out_k.size(2) == dim && out_v.size(2) == dim, "dim mismatch");
    TORCH_CHECK(m >= 0 && m <= indices.size(1) && m <= out_k.size(1) && m <= out_v.size(1),
                "m exceeds buffer capacity");
    TORCH_CHECK((2 * dim * static_cast<int64_t>(sizeof(nv_bfloat16))) % sizeof(uint4) == 0,
                "2*dim*2B must be divisible by 16 for vectorized UVA reads");
    if (rows == 0 || m == 0) {
        return;
    }
    constexpr int kWarpsPerBlock = 8;
    dim3 grid((m + kWarpsPerBlock - 1) / kWarpsPerBlock, rows);
    auto stream = at::cuda::getCurrentCUDAStream();
    cometkv::uva_gather_kv_rows_kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        indices.data_ptr<int32_t>(),
        reinterpret_cast<const nv_bfloat16*>(cpu_kv.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(out_k.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(out_v.data_ptr()),
        static_cast<int>(m),
        cpu_kv.size(1),
        static_cast<int>(dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void uva_gather_kv_rows_window(
    torch::Tensor indices,   // [rows, m_cap] int32 (device)
    torch::Tensor kv_ptrs,   // [n_layers] int64 (device): pinned-store data ptrs, 0 = skip
    torch::Tensor out_k,     // [n_layers, rows, m_cap, dim] bf16 (device)
    torch::Tensor out_v,
    int64_t m,
    int64_t total_tokens) {
    check_int32_tensor(indices, "indices");
    check_cuda_tensor(kv_ptrs, "kv_ptrs");
    check_cuda_tensor(out_k, "out_k");
    check_cuda_tensor(out_v, "out_v");
    TORCH_CHECK(kv_ptrs.scalar_type() == torch::kInt64, "kv_ptrs must be int64");
    TORCH_CHECK(out_k.scalar_type() == torch::kBFloat16 && out_v.scalar_type() == torch::kBFloat16,
                "out_k/out_v must be bf16");
    TORCH_CHECK(indices.dim() == 2 && out_k.dim() == 4 && out_v.dim() == 4 && kv_ptrs.dim() == 1,
                "shape ranks mismatch");
    TORCH_CHECK(indices.is_contiguous() && out_k.is_contiguous() && out_v.is_contiguous()
                && kv_ptrs.is_contiguous(), "all tensors must be contiguous");
    const int64_t n_layers = out_k.size(0);
    const int64_t rows = indices.size(0);
    const int64_t dim = out_k.size(3);
    TORCH_CHECK(kv_ptrs.size(0) == n_layers && out_v.sizes() == out_k.sizes(), "window shape mismatch");
    TORCH_CHECK(out_k.size(1) == rows, "rows mismatch");
    TORCH_CHECK(m >= 0 && m <= indices.size(1) && m <= out_k.size(2), "m exceeds buffer capacity");
    TORCH_CHECK((2 * dim * static_cast<int64_t>(sizeof(nv_bfloat16))) % sizeof(uint4) == 0,
                "2*dim*2B must be divisible by 16 for vectorized UVA reads");
    if (rows == 0 || m == 0 || n_layers == 0) {
        return;
    }
    constexpr int kWarpsPerBlock = 8;
    dim3 grid((m + kWarpsPerBlock - 1) / kWarpsPerBlock, rows, n_layers);
    auto stream = at::cuda::getCurrentCUDAStream();
    cometkv::uva_gather_kv_rows_window_kernel<<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        indices.data_ptr<int32_t>(),
        kv_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<nv_bfloat16*>(out_k.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(out_v.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(m),
        total_tokens,
        static_cast<int>(dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("asym_signature_score_into", &asym_signature_score_into,
          "Asymmetric signature scoring into a caller-owned [rows, tokens] fp32 buffer (CUDA)",
          py::arg("query_proj"), py::arg("query_proj_total"), py::arg("keys"),
          py::arg("range_starts"), py::arg("range_ends"), py::arg("scores"),
          py::arg("sig_bits_used"), py::arg("active_candidates"),
          py::arg("norm_lo") = py::none(), py::arg("norm_step") = py::none());
    m.def("sampled_tail_attention_merge", &sampled_tail_attention_merge,
          "Fused importance-corrected micro-attention over sampled tail tokens, LSE-merged "
          "in place into the main flash output (CUDA); clip>0 caps logits at block mean+clip",
          py::arg("queries"), py::arg("tail_keys"), py::arg("tail_values"), py::arg("corr"),
          py::arg("out_main"), py::arg("lse_main"), py::arg("scale"), py::arg("clip"));
    m.def("uva_gather_kv_rows", &uva_gather_kv_rows,
          "Cache-bypass UVA gather of sampled tail K/V rows from the pinned host KV store (CUDA)",
          py::arg("indices"), py::arg("cpu_kv"), py::arg("out_k"), py::arg("out_v"), py::arg("m"));
    m.def("uva_gather_kv_rows_window", &uva_gather_kv_rows_window,
          "Batched multi-layer cache-bypass UVA tail gather (one launch per resample window)",
          py::arg("indices"), py::arg("kv_ptrs"), py::arg("out_k"), py::arg("out_v"),
          py::arg("m"), py::arg("total_tokens"));
}
