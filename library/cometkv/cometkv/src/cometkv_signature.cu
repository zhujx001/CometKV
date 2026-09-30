#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <vector>

#include "cometkv_signature_kernel.cuh"
#include "cometkv_topk_kernel.cuh"

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


void grouped_signature_score_into(
    torch::Tensor query_proj, torch::Tensor query_total, torch::Tensor keys,
    torch::Tensor range_starts, torch::Tensor range_ends,
    torch::Tensor norm_lo, torch::Tensor norm_step, torch::Tensor center_bias,
    torch::Tensor head_logits, torch::Tensor chunk_lse, torch::Tensor head_lse,
    torch::Tensor scores, int64_t sig_bits_used, int64_t active_candidates,
    int64_t prompt_length, int64_t indexed_length, int64_t block_size, int64_t overlap,
    double residual_scale, double bias_scale, bool normalize_heads, bool use_lookup) {
    check_cuda_tensor(keys, "keys");
    const c10::cuda::CUDAGuard device_guard(keys.device());
    TORCH_CHECK(keys.dim() == 3 && keys.scalar_type() == torch::kUInt8,
                "keys must be uint8 [rows, tokens, sig_bytes]");
    check_cuda_tensor(query_proj, "query_proj");
    TORCH_CHECK(query_proj.dim() == 3, "query_proj must be [rows, group, width]");
    const int64_t rows = keys.size(0), tokens = keys.size(1), bytes = keys.size(2);
    const int64_t group = query_proj.size(1), width = query_proj.size(2);
    TORCH_CHECK(norm_lo.dim() == 2, "norm_lo must be [rows, blocks]");
    const int64_t blocks = norm_lo.size(1);
    const int64_t chunk_capacity = (tokens + 255) / 256;
    for (const auto& item : std::vector<std::pair<torch::Tensor, const char*>>{
             {query_proj, "query_proj"}, {query_total, "query_total"},
             {norm_lo, "norm_lo"}, {norm_step, "norm_step"}, {center_bias, "center_bias"},
             {head_logits, "head_logits"}, {chunk_lse, "chunk_lse"},
             {head_lse, "head_lse"}, {scores, "scores"}}) {
        check_cuda_tensor(item.first, item.second);
        TORCH_CHECK(item.first.scalar_type() == torch::kFloat32, item.second, " must be fp32");
        TORCH_CHECK(item.first.device() == keys.device(), item.second, " must be on the keys device");
    }
    check_int32_tensor(range_starts, "range_starts");
    check_int32_tensor(range_ends, "range_ends");
    TORCH_CHECK(range_starts.device() == keys.device() && range_ends.device() == keys.device(),
                "ranges must be on the keys device");
    TORCH_CHECK(group > 0 && query_proj.size(0) == rows, "query group/rows mismatch");
    TORCH_CHECK(sig_bits_used > 0 && sig_bits_used <= 128 && sig_bits_used % 8 == 0
                && sig_bits_used <= width && sig_bits_used / 8 < bytes,
                "invalid signature dimensions: reserve one byte for the norm");
    TORCH_CHECK(query_total.sizes().vec() == std::vector<int64_t>({rows, group}), "query_total shape mismatch");
    TORCH_CHECK(norm_lo.size(0) == rows && blocks > 0 && norm_step.sizes() == norm_lo.sizes(), "norm shapes mismatch");
    TORCH_CHECK(center_bias.sizes().vec() == std::vector<int64_t>({rows, group, blocks}), "center_bias shape mismatch");
    TORCH_CHECK(head_logits.sizes().vec() == std::vector<int64_t>({rows, group, tokens}), "head_logits shape mismatch");
    TORCH_CHECK(chunk_lse.sizes().vec() == std::vector<int64_t>({rows, group, chunk_capacity}), "chunk_lse shape mismatch");
    TORCH_CHECK(head_lse.sizes().vec() == std::vector<int64_t>({rows, group}), "head_lse shape mismatch");
    TORCH_CHECK(scores.sizes().vec() == std::vector<int64_t>({rows, tokens}), "scores shape mismatch");
    TORCH_CHECK(range_starts.sizes().vec() == std::vector<int64_t>({rows, 2})
                && range_ends.sizes() == range_starts.sizes(), "ranges must be [rows, 2]");
    TORCH_CHECK(prompt_length >= 0 && prompt_length <= tokens && block_size > 0 && overlap >= 0,
                "invalid block layout");
    TORCH_CHECK(indexed_length >= 0 && indexed_length <= tokens, "invalid indexed_length");
    const int64_t required_blocks = indexed_length > prompt_length
        ? 2 + (indexed_length - prompt_length - 1 + overlap) / block_size : 1;
    TORCH_CHECK(blocks >= required_blocks, "block metadata does not cover signature capacity");
    TORCH_CHECK(normalize_heads || group == 1, "unnormalized scoring requires one query group");
    const int64_t candidates = active_candidates < 0 ? tokens : std::min(active_candidates, tokens);
    const int64_t chunks = (candidates + 255) / 256;
    if (!rows || !chunks) return;
    dim3 grid(rows, chunks, (group + 3) / 4);
    auto stream = at::cuda::getCurrentCUDAStream();
    // Unnormalized group=1 writes directly into the final score buffer.
    float* logit_ptr = normalize_heads ? head_logits.data_ptr<float>() : scores.data_ptr<float>();
    #define COMETKV_GROUPED_ARGS \
        query_proj.data_ptr<float>(), query_total.data_ptr<float>(), keys.data_ptr<uint8_t>(), \
        range_starts.data_ptr<int32_t>(), range_ends.data_ptr<int32_t>(), \
        norm_lo.data_ptr<float>(), norm_step.data_ptr<float>(), center_bias.data_ptr<float>(), \
        logit_ptr, chunk_lse.data_ptr<float>(), static_cast<int>(tokens), static_cast<int>(bytes), \
        static_cast<int>(sig_bits_used), static_cast<int>(width), static_cast<int>(group), \
        static_cast<int>(blocks), static_cast<int>(chunk_capacity), static_cast<int>(prompt_length), \
        static_cast<int>(block_size), static_cast<int>(overlap), \
        static_cast<float>(residual_scale), static_cast<float>(bias_scale)
    if (normalize_heads) {
        if (use_lookup)
            cometkv::grouped_signature_logits_kernel<true, true><<<grid, 256, 0, stream>>>(COMETKV_GROUPED_ARGS);
        else
            cometkv::grouped_signature_logits_kernel<true><<<grid, 256, 0, stream>>>(COMETKV_GROUPED_ARGS);
        cometkv::grouped_signature_lse_kernel<<<rows * group, 256, 0, stream>>>(
            chunk_lse.data_ptr<float>(), head_lse.data_ptr<float>(), group, chunks, chunk_capacity);
        cometkv::grouped_signature_merge_kernel<<<dim3(rows, chunks), 256, 0, stream>>>(
            head_logits.data_ptr<float>(), head_lse.data_ptr<float>(),
            range_starts.data_ptr<int32_t>(), range_ends.data_ptr<int32_t>(),
            scores.data_ptr<float>(), group, tokens);
    } else {
        if (use_lookup)
            cometkv::grouped_signature_logits_kernel<false, true><<<grid, 256, 0, stream>>>(COMETKV_GROUPED_ARGS);
        else
            cometkv::grouped_signature_logits_kernel<false><<<grid, 256, 0, stream>>>(COMETKV_GROUPED_ARGS);
    }
    #undef COMETKV_GROUPED_ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void sampled_tail_attention_merge(
    torch::Tensor queries,      // [rows, group, dim] bf16
    torch::Tensor tail_keys,    // [rows, m, dim] bf16
    torch::Tensor tail_values,  // [rows, m, dim] bf16
    torch::Tensor corr,         // [rows, m] fp32
    torch::Tensor out_main,     // [rows, group, dim] bf16 (in/out)
    torch::Tensor lse_main,     // [rows, group] fp32
    double scale,
    double clip,
    c10::optional<torch::Tensor> sample_indices,
    c10::optional<torch::Tensor> head_indices,
    int64_t head_len) {
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
    TORCH_CHECK(sample_indices.has_value() == head_indices.has_value(),
                "sample_indices and head_indices must be provided together");
    const int32_t* sample_ptr = nullptr;
    const int32_t* head_ptr = nullptr;
    int64_t head_stride = 0;
    if (sample_indices.has_value()) {
        const auto& samples = sample_indices.value();
        const auto& heads = head_indices.value();
        check_cuda_tensor(samples, "sample_indices");
        check_cuda_tensor(heads, "head_indices");
        TORCH_CHECK(samples.device() == queries.device() && heads.device() == queries.device(),
                    "sample/head indices must be on the query device");
        TORCH_CHECK(samples.scalar_type() == torch::kInt32 && heads.scalar_type() == torch::kInt32,
                    "sample/head indices must be int32");
        TORCH_CHECK(samples.is_contiguous() && heads.is_contiguous(), "indices must be contiguous");
        TORCH_CHECK(samples.dim() == 2 && samples.size(0) == rows && samples.size(1) == m,
                    "sample_indices must be [rows, m]");
        TORCH_CHECK(heads.dim() == 2 && heads.size(0) == rows, "head_indices must be [rows, capacity]");
        head_stride = heads.size(1);
        if (head_len == -1) head_len = head_stride;
        TORCH_CHECK(head_len >= 0 && head_len <= head_stride, "invalid head_len");
        sample_ptr = samples.data_ptr<int32_t>();
        head_ptr = heads.data_ptr<int32_t>();
    } else {
        TORCH_CHECK(head_len == -1 || head_len == 0, "head_len requires indices");
        head_len = 0;
    }
    if (rows == 0 || m == 0) {
        return;
    }
    size_t smem = static_cast<size_t>((dim + m + 3) & ~int64_t(3)) * sizeof(float);
    constexpr int threads = 256;
    constexpr size_t shared_limit = 48 * 1024 - 512;  // Includes static reduction storage/alignment.
    constexpr size_t vector_scratch = (threads / 32) * 128 * sizeof(float);
    const bool vectorized = dim == 128 && smem + vector_scratch <= shared_limit
        && reinterpret_cast<uintptr_t>(tail_keys.data_ptr()) % alignof(uint2) == 0
        && reinterpret_cast<uintptr_t>(tail_values.data_ptr()) % alignof(uint2) == 0;
    const size_t value_scratch = vectorized ? vector_scratch : 0;
    TORCH_CHECK(smem <= shared_limit, "sampled tail m too large for shared memory");
    int hash_capacity = 0;
    if (head_len > 0 && head_len <= 4096) {
        hash_capacity = 32;
        while (hash_capacity < 2 * head_len) hash_capacity *= 2;
        if (smem + hash_capacity * sizeof(int32_t) > shared_limit) hash_capacity = 0;
    }
    smem += std::max(hash_capacity * sizeof(int32_t), value_scratch);
    dim3 grid(rows, group);
    auto stream = at::cuda::getCurrentCUDAStream();
    cometkv::sampled_tail_attention_merge_kernel<threads><<<grid, threads, smem, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(queries.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(tail_keys.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(tail_values.data_ptr()),
        corr.data_ptr<float>(),
        sample_ptr, head_ptr, static_cast<int>(head_len), static_cast<int>(head_stride),
        hash_capacity, vectorized,
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


void exact_topk_indices_into(
    torch::Tensor scores, torch::Tensor indices, torch::Tensor histogram,
    torch::Tensor candidates, torch::Tensor state, int64_t k, int64_t start, int64_t end) {
    check_cuda_tensor(scores, "scores");
    const c10::cuda::CUDAGuard guard(scores.device());
    TORCH_CHECK(scores.dim() == 2 && scores.scalar_type() == torch::kFloat32,
                "scores must be fp32 [rows, tokens]");
    const int64_t rows = scores.size(0), tokens = scores.size(1);
    for (const auto& item : std::vector<std::pair<torch::Tensor, const char*>>{
             {indices, "indices"}, {histogram, "histogram"},
             {candidates, "candidates"}, {state, "state"}}) {
        check_int32_tensor(item.first, item.second);
        TORCH_CHECK(item.first.device() == scores.device(), item.second, " must be on the score device");
    }
    TORCH_CHECK(start >= 0 && start <= end && end <= tokens, "invalid candidate range");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == rows, "indices must be [rows, capacity]");
    TORCH_CHECK(k >= 0 && k <= end - start && k <= indices.size(1) && k <= 4096, "invalid k (max 4096)");
    const int64_t chunks = (end - start + cometkv::kSelectChunk - 1) / cometkv::kSelectChunk;
    TORCH_CHECK(histogram.dim() == 3 && histogram.size(0) == rows && histogram.size(1) >= chunks
                && histogram.size(2) == cometkv::kSelectBinsStride,
                "histogram must be [rows, chunk_capacity, 288]");
    TORCH_CHECK(candidates.sizes() == scores.sizes(), "candidates must match scores shape");
    TORCH_CHECK(state.sizes().vec() == std::vector<int64_t>({rows, cometkv::kSelectStateSize}),
                "state must be [rows, 7]");
    if (!rows || !k) return;
    const float* input = scores.data_ptr<float>();
    int* output = indices.data_ptr<int32_t>();
    int* hist = histogram.data_ptr<int32_t>();
    int* work = candidates.data_ptr<int32_t>();
    int* params = state.data_ptr<int32_t>();
    const int stride = indices.size(1), hist_stride = histogram.size(1);
    const dim3 grid(rows, chunks);
    auto stream = at::cuda::getCurrentCUDAStream();
    cometkv::topk_histogram_kernel<0><<<grid, 256, 0, stream>>>(input, hist, params, tokens, start, end, hist_stride);
    cometkv::topk_choose_prefix_kernel<0><<<rows, 256, 0, stream>>>(hist, params, chunks, hist_stride, k);
    cometkv::topk_histogram_kernel<1><<<grid, 256, 0, stream>>>(input, hist, params, tokens, start, end, hist_stride);
    cometkv::topk_choose_prefix_kernel<1><<<rows, 256, 0, stream>>>(hist, params, chunks, hist_stride, k);
    cometkv::topk_compact_kernel<<<grid, 256, 0, stream>>>(input, output, work, params, tokens, stride, start, end);
    #define COMETKV_REFINE(ITEMS) \
        cometkv::topk_refine_kernel<ITEMS><<<rows, 256, 0, stream>>>(input, output, work, params, tokens, stride, k, start)
    if (k <= 256) { COMETKV_REFINE(1); }
    else if (k <= 512) { COMETKV_REFINE(2); }
    else if (k <= 1024) { COMETKV_REFINE(4); }
    else if (k <= 2048) { COMETKV_REFINE(8); }
    else { COMETKV_REFINE(16); }
    #undef COMETKV_REFINE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exact_topk_indices_into", &exact_topk_indices_into,
          "Exact largest-k selection into int32 indices (ascending token order, lower token ID wins ties)");
    m.def("grouped_signature_score_into", &grouped_signature_score_into,
          "Block-compensated GQA scoring; optionally log-mean of per-head softmax probabilities (CUDA)",
          py::arg("query_proj"), py::arg("query_total"), py::arg("keys"),
          py::arg("range_starts"), py::arg("range_ends"), py::arg("norm_lo"),
          py::arg("norm_step"), py::arg("center_bias"), py::arg("head_logits"),
          py::arg("chunk_lse"), py::arg("head_lse"), py::arg("scores"),
          py::arg("sig_bits_used"), py::arg("active_candidates"), py::arg("prompt_length"),
          py::arg("indexed_length"), py::arg("block_size"), py::arg("overlap"),
          py::arg("residual_scale"), py::arg("bias_scale"), py::arg("normalize_heads"),
          py::arg("use_lookup") = false);
    m.def("asym_signature_score_into", &asym_signature_score_into,
          "Asymmetric signature scoring into a caller-owned [rows, tokens] fp32 buffer (CUDA)",
          py::arg("query_proj"), py::arg("query_proj_total"), py::arg("keys"),
          py::arg("range_starts"), py::arg("range_ends"), py::arg("scores"),
          py::arg("sig_bits_used"), py::arg("active_candidates"),
          py::arg("norm_lo") = py::none(), py::arg("norm_step") = py::none());
    m.def("sampled_tail_attention_merge", &sampled_tail_attention_merge,
          "Fused importance-corrected micro-attention over sampled tail tokens, LSE-merged "
          "in place into the main flash output; optional indices exclude current-head samples",
          py::arg("queries"), py::arg("tail_keys"), py::arg("tail_values"), py::arg("corr"),
          py::arg("out_main"), py::arg("lse_main"), py::arg("scale"), py::arg("clip"),
          py::arg("sample_indices") = py::none(), py::arg("head_indices") = py::none(),
          py::arg("head_len") = -1);
    m.def("uva_gather_kv_rows", &uva_gather_kv_rows,
          "Cache-bypass UVA gather of sampled tail K/V rows from the pinned host KV store (CUDA)",
          py::arg("indices"), py::arg("cpu_kv"), py::arg("out_k"), py::arg("out_v"), py::arg("m"));
    m.def("uva_gather_kv_rows_window", &uva_gather_kv_rows_window,
          "Batched multi-layer cache-bypass UVA tail gather (one launch per resample window)",
          py::arg("indices"), py::arg("kv_ptrs"), py::arg("out_k"), py::arg("out_v"),
          py::arg("m"), py::arg("total_tokens"));
}
