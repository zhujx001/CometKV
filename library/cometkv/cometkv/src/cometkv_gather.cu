#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include "cometkv_gather_kernel.cuh"

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cpu_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.is_pinned(), name, " must be pinned for UVA access");
}

void check_two_byte_dtype(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.element_size() == 2, name, " must use a 2-byte dtype");
}

void launch_append_lockstep_local_kv_cache(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor local_keys,
    torch::Tensor local_values,
    const torch::Tensor* visible_lengths,
    int64_t decode_step,
    int64_t local_capacity,
    int64_t slide_stride,
    const torch::Tensor* decode_step_dev = nullptr) {
    check_cuda_tensor(key_states, "key_states");
    check_cuda_tensor(value_states, "value_states");
    check_cuda_tensor(local_keys, "local_keys");
    check_cuda_tensor(local_values, "local_values");
    check_two_byte_dtype(key_states, "key_states");
    check_two_byte_dtype(value_states, "value_states");
    check_two_byte_dtype(local_keys, "local_keys");
    check_two_byte_dtype(local_values, "local_values");
    if (visible_lengths != nullptr) {
        check_cuda_tensor(*visible_lengths, "visible_lengths");
        TORCH_CHECK(visible_lengths->scalar_type() == torch::kInt32, "visible_lengths must be int32");
        TORCH_CHECK(visible_lengths->dim() == 1, "visible_lengths must be [batch]");
    }

    TORCH_CHECK(key_states.dim() == 4, "key_states must be [batch, 1, kv_heads, dim]");
    TORCH_CHECK(value_states.dim() == 4, "value_states must be [batch, 1, kv_heads, dim]");
    TORCH_CHECK(local_keys.dim() == 4, "local_keys must be [batch, kv_heads, local_tokens, dim]");
    TORCH_CHECK(local_values.dim() == 4, "local_values must be [batch, kv_heads, local_tokens, dim]");

    const int batch_size = static_cast<int>(key_states.size(0));
    const int seq_len = static_cast<int>(key_states.size(1));
    const int kv_heads = static_cast<int>(key_states.size(2));
    const int dim = static_cast<int>(key_states.size(3));
    const int local_tokens = static_cast<int>(local_keys.size(2));

    TORCH_CHECK(seq_len == 1, "key_states seq_len must be 1");
    TORCH_CHECK(value_states.size(0) == batch_size && value_states.size(1) == 1 &&
                value_states.size(2) == kv_heads && value_states.size(3) == dim,
                "value_states shape mismatch");
    TORCH_CHECK(local_keys.size(0) == batch_size && local_keys.size(1) == kv_heads &&
                local_keys.size(3) == dim, "local_keys shape mismatch");
    TORCH_CHECK(local_values.size(0) == batch_size && local_values.size(1) == kv_heads &&
                local_values.size(2) == local_tokens && local_values.size(3) == dim,
                "local_values shape mismatch");
    if (visible_lengths != nullptr) {
        TORCH_CHECK(visible_lengths->size(0) == batch_size, "visible_lengths shape mismatch");
    }
    TORCH_CHECK(decode_step >= 0, "decode_step must be non-negative");
    TORCH_CHECK(local_capacity > 0, "local_capacity must be positive");
    TORCH_CHECK(slide_stride > 0, "slide_stride must be positive");
    TORCH_CHECK(local_capacity <= local_tokens, "local_capacity cannot exceed local token storage");

    int32_t* visible_lengths_ptr = nullptr;
    if (visible_lengths != nullptr) {
        visible_lengths_ptr = visible_lengths->data_ptr<int32_t>();
    }
    const int32_t* decode_step_dev_ptr = nullptr;
    if (decode_step_dev != nullptr) {
        check_cuda_tensor(*decode_step_dev, "decode_step_dev");
        TORCH_CHECK(decode_step_dev->scalar_type() == torch::kInt32, "decode_step_dev must be int32");
        decode_step_dev_ptr = decode_step_dev->data_ptr<int32_t>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(batch_size * kv_heads);
    if (visible_lengths_ptr != nullptr) {
        cometkv::append_lockstep_local_kv_cache_kernel<true><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(key_states.data_ptr()),
            reinterpret_cast<const uint16_t*>(value_states.data_ptr()),
            reinterpret_cast<uint16_t*>(local_keys.data_ptr()),
            reinterpret_cast<uint16_t*>(local_values.data_ptr()),
            visible_lengths_ptr,
            batch_size,
            kv_heads,
            local_tokens,
            static_cast<int>(local_capacity),
            static_cast<int>(slide_stride),
            static_cast<int>(decode_step),
            dim,
            decode_step_dev_ptr
        );
    } else {
        cometkv::append_lockstep_local_kv_cache_kernel<false><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(key_states.data_ptr()),
            reinterpret_cast<const uint16_t*>(value_states.data_ptr()),
            reinterpret_cast<uint16_t*>(local_keys.data_ptr()),
            reinterpret_cast<uint16_t*>(local_values.data_ptr()),
            nullptr,
            batch_size,
            kv_heads,
            local_tokens,
            static_cast<int>(local_capacity),
            static_cast<int>(slide_stride),
            static_cast<int>(decode_step),
            dim,
            decode_step_dev_ptr
        );
    }
}

}  // namespace



void append_lockstep_local_kv_cache(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor local_keys,
    torch::Tensor local_values,
    int64_t decode_step,
    int64_t local_capacity,
    int64_t slide_stride) {
    launch_append_lockstep_local_kv_cache(
        key_states,
        value_states,
        local_keys,
        local_values,
        nullptr,
        decode_step,
        local_capacity,
        slide_stride
    );
}

void append_lockstep_local_kv_cache_and_advance(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor local_keys,
    torch::Tensor local_values,
    torch::Tensor visible_lengths,
    int64_t decode_step,
    int64_t local_capacity,
    int64_t slide_stride) {
    launch_append_lockstep_local_kv_cache(
        key_states,
        value_states,
        local_keys,
        local_values,
        &visible_lengths,
        decode_step,
        local_capacity,
        slide_stride
    );
}

void append_lockstep_local_kv_cache_dev(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor local_keys,
    torch::Tensor local_values,
    torch::Tensor decode_step_dev,
    int64_t local_capacity,
    int64_t slide_stride) {
    // CUDA-graph capturable append (no visible_lengths advance) reading decode_step from a device counter.
    launch_append_lockstep_local_kv_cache(
        key_states, value_states, local_keys, local_values,
        nullptr, 0, local_capacity, slide_stride, &decode_step_dev);
}

void refresh_static_prompt_recent_state(
    torch::Tensor prompt_lengths,
    torch::Tensor visible_lengths,
    torch::Tensor prompt_recent_keys,
    torch::Tensor prompt_recent_values,
    torch::Tensor static_keys,
    torch::Tensor static_values,
    torch::Tensor static_lengths,
    int64_t static_pattern_start,
    int64_t static_pattern_end) {
    check_cuda_tensor(prompt_lengths, "prompt_lengths");
    check_cuda_tensor(visible_lengths, "visible_lengths");
    check_cuda_tensor(prompt_recent_keys, "prompt_recent_keys");
    check_cuda_tensor(prompt_recent_values, "prompt_recent_values");
    check_cuda_tensor(static_keys, "static_keys");
    check_cuda_tensor(static_values, "static_values");
    check_cuda_tensor(static_lengths, "static_lengths");
    check_two_byte_dtype(prompt_recent_keys, "prompt_recent_keys");
    check_two_byte_dtype(prompt_recent_values, "prompt_recent_values");
    check_two_byte_dtype(static_keys, "static_keys");
    check_two_byte_dtype(static_values, "static_values");

    TORCH_CHECK(prompt_lengths.scalar_type() == torch::kInt32, "prompt_lengths must be int32");
    TORCH_CHECK(visible_lengths.scalar_type() == torch::kInt32, "visible_lengths must be int32");
    TORCH_CHECK(static_lengths.scalar_type() == torch::kInt32, "static_lengths must be int32");
    TORCH_CHECK(prompt_lengths.dim() == 1, "prompt_lengths must be [batch]");
    TORCH_CHECK(visible_lengths.dim() == 1, "visible_lengths must be [batch]");
    TORCH_CHECK(prompt_recent_keys.dim() == 4 && prompt_recent_values.dim() == 4,
                "prompt recent tensors must be [batch, kv_heads, static_pattern_end, dim]");
    TORCH_CHECK(static_keys.dim() == 4 && static_values.dim() == 4,
                "static tensors must be [batch, kv_heads, static_pattern_total, dim]");
    TORCH_CHECK(static_lengths.dim() == 1, "static_lengths must be [batch * kv_heads]");

    const int batch_size = static_cast<int>(prompt_recent_keys.size(0));
    const int kv_heads = static_cast<int>(prompt_recent_keys.size(1));
    const int recent_tokens = static_cast<int>(prompt_recent_keys.size(2));
    const int dim = static_cast<int>(prompt_recent_keys.size(3));
    const int static_total = static_cast<int>(static_pattern_start + static_pattern_end);

    TORCH_CHECK(prompt_lengths.size(0) == batch_size, "prompt_lengths shape mismatch");
    TORCH_CHECK(visible_lengths.size(0) == batch_size, "visible_lengths shape mismatch");
    TORCH_CHECK(prompt_recent_values.sizes().vec() == prompt_recent_keys.sizes().vec(), "prompt recent value shape mismatch");
    TORCH_CHECK(recent_tokens == static_pattern_end, "prompt recent length mismatch");
    TORCH_CHECK(static_keys.size(0) == batch_size && static_keys.size(1) == kv_heads &&
                static_keys.size(2) == static_total && static_keys.size(3) == dim,
                "static_keys shape mismatch");
    TORCH_CHECK(static_values.sizes().vec() == static_keys.sizes().vec(), "static_values shape mismatch");
    TORCH_CHECK(static_lengths.size(0) == batch_size * kv_heads, "static_lengths shape mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(batch_size * kv_heads);
    cometkv::refresh_static_prompt_recent_state_kernel<<<grid, 128, 0, stream>>>(
        prompt_lengths.data_ptr<int32_t>(),
        visible_lengths.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(prompt_recent_keys.data_ptr()),
        reinterpret_cast<const uint16_t*>(prompt_recent_values.data_ptr()),
        reinterpret_cast<uint16_t*>(static_keys.data_ptr()),
        reinterpret_cast<uint16_t*>(static_values.data_ptr()),
        static_lengths.data_ptr<int32_t>(),
        batch_size,
        kv_heads,
        static_cast<int>(static_pattern_start),
        static_cast<int>(static_pattern_end),
        dim
    );
}


void copy_interleaved_kv_to_cpu(
    torch::Tensor key_rows,
    torch::Tensor value_rows,
    torch::Tensor cpu_kv,
    int64_t row_start,
    int64_t token_start) {
    check_cuda_tensor(key_rows, "key_rows");
    check_cuda_tensor(value_rows, "value_rows");
    check_cpu_tensor(cpu_kv, "cpu_kv");
    check_two_byte_dtype(key_rows, "key_rows");
    check_two_byte_dtype(value_rows, "value_rows");
    check_two_byte_dtype(cpu_kv, "cpu_kv");

    TORCH_CHECK(key_rows.scalar_type() == value_rows.scalar_type(), "key/value dtype mismatch");
    TORCH_CHECK(key_rows.scalar_type() == cpu_kv.scalar_type(), "cpu_kv dtype mismatch");
    TORCH_CHECK(key_rows.dim() == 3, "key_rows must be [rows, tokens, dim]");
    TORCH_CHECK(value_rows.sizes().vec() == key_rows.sizes().vec(), "value_rows shape mismatch");
    TORCH_CHECK(cpu_kv.dim() == 4, "cpu_kv must be [rows, total_tokens, 2, dim]");
    TORCH_CHECK(cpu_kv.size(2) == 2, "cpu_kv must have K/V dimension size 2");

    const int64_t rows = key_rows.size(0);
    const int64_t tokens = key_rows.size(1);
    const int64_t dim = key_rows.size(2);
    const int64_t total_rows = cpu_kv.size(0);
    const int64_t total_tokens = cpu_kv.size(1);

    TORCH_CHECK(cpu_kv.size(3) == dim, "cpu_kv dim mismatch");
    TORCH_CHECK(row_start >= 0 && token_start >= 0, "row_start/token_start must be non-negative");
    TORCH_CHECK(row_start + rows <= total_rows, "row range exceeds cpu_kv rows");
    TORCH_CHECK(token_start + tokens <= total_tokens, "token range exceeds cpu_kv tokens");

    c10::cuda::CUDAGuard device_guard(key_rows.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const size_t elem_bytes = static_cast<size_t>(key_rows.element_size());
    const size_t width_bytes = static_cast<size_t>(dim) * elem_bytes;
    const size_t src_pitch_bytes = width_bytes;
    const size_t dst_pitch_bytes = static_cast<size_t>(2 * dim) * elem_bytes;
    const size_t cpu_row_pitch_bytes = static_cast<size_t>(total_tokens * 2 * dim) * elem_bytes;
    const size_t gpu_row_pitch_bytes = static_cast<size_t>(tokens * dim) * elem_bytes;

    const char* key_base = static_cast<const char*>(key_rows.data_ptr());
    const char* value_base = static_cast<const char*>(value_rows.data_ptr());
    char* cpu_base = static_cast<char*>(cpu_kv.data_ptr());

    for (int64_t row = 0; row < rows; ++row) {
        char* dst_row = cpu_base
            + static_cast<size_t>(row_start + row) * cpu_row_pitch_bytes
            + static_cast<size_t>(token_start * 2 * dim) * elem_bytes;
        const char* src_key = key_base + static_cast<size_t>(row) * gpu_row_pitch_bytes;
        const char* src_value = value_base + static_cast<size_t>(row) * gpu_row_pitch_bytes;

        C10_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_row,
            dst_pitch_bytes,
            src_key,
            src_pitch_bytes,
            width_bytes,
            static_cast<size_t>(tokens),
            cudaMemcpyDeviceToHost,
            stream
        ));
        C10_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_row + width_bytes,
            dst_pitch_bytes,
            src_value,
            src_pitch_bytes,
            width_bytes,
            static_cast<size_t>(tokens),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }
}


void copy_interleaved_int8_kv_to_cpu(
    torch::Tensor key_codes,
    torch::Tensor value_codes,
    torch::Tensor value_scale,
    torch::Tensor cpu_kv_int8,
    torch::Tensor cpu_value_scale,
    int64_t row_start,
    int64_t token_start) {
    check_cuda_tensor(key_codes, "key_codes");
    check_cuda_tensor(value_codes, "value_codes");
    check_cuda_tensor(value_scale, "value_scale");
    check_cpu_tensor(cpu_kv_int8, "cpu_kv_int8");
    check_cpu_tensor(cpu_value_scale, "cpu_value_scale");

    TORCH_CHECK(key_codes.scalar_type() == torch::kChar, "key_codes must be int8");
    TORCH_CHECK(value_codes.scalar_type() == torch::kChar, "value_codes must be int8");
    TORCH_CHECK(value_scale.scalar_type() == torch::kFloat32, "value_scale must be float32");
    TORCH_CHECK(cpu_kv_int8.scalar_type() == torch::kChar, "cpu_kv_int8 must be int8");
    TORCH_CHECK(cpu_value_scale.scalar_type() == torch::kFloat32, "cpu_value_scale must be float32");

    TORCH_CHECK(key_codes.dim() == 3, "key_codes must be [rows, tokens, dim]");
    TORCH_CHECK(value_codes.sizes().vec() == key_codes.sizes().vec(), "value_codes shape mismatch");
    TORCH_CHECK(value_scale.dim() == 2, "value_scale must be [rows, tokens]");
    TORCH_CHECK(value_scale.size(0) == key_codes.size(0) && value_scale.size(1) == key_codes.size(1),
                "value_scale shape mismatch");
    TORCH_CHECK(cpu_kv_int8.dim() == 4, "cpu_kv_int8 must be [rows, total_tokens, 2, dim]");
    TORCH_CHECK(cpu_kv_int8.size(2) == 2, "cpu_kv_int8 must have K/V dimension size 2");
    TORCH_CHECK(cpu_value_scale.dim() == 2, "cpu_value_scale must be [rows, total_tokens]");

    const int64_t rows = key_codes.size(0);
    const int64_t tokens = key_codes.size(1);
    const int64_t dim = key_codes.size(2);
    const int64_t total_rows = cpu_kv_int8.size(0);
    const int64_t total_tokens = cpu_kv_int8.size(1);

    TORCH_CHECK(cpu_kv_int8.size(3) == dim, "cpu_kv_int8 dim mismatch");
    TORCH_CHECK(cpu_value_scale.size(0) == total_rows && cpu_value_scale.size(1) == total_tokens,
                "cpu_value_scale shape mismatch");
    TORCH_CHECK(row_start >= 0 && token_start >= 0, "row_start/token_start must be non-negative");
    TORCH_CHECK(row_start + rows <= total_rows, "row range exceeds cpu storage rows");
    TORCH_CHECK(token_start + tokens <= total_tokens, "token range exceeds cpu storage tokens");

    c10::cuda::CUDAGuard device_guard(key_codes.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const size_t code_width_bytes = static_cast<size_t>(dim);
    const size_t code_src_pitch_bytes = code_width_bytes;
    const size_t code_dst_pitch_bytes = static_cast<size_t>(2 * dim);
    const size_t cpu_kv_row_pitch_bytes = static_cast<size_t>(total_tokens * 2 * dim);
    const size_t gpu_code_row_pitch_bytes = static_cast<size_t>(tokens * dim);

    const char* key_base = static_cast<const char*>(key_codes.data_ptr());
    const char* value_base = static_cast<const char*>(value_codes.data_ptr());
    char* cpu_kv_base = static_cast<char*>(cpu_kv_int8.data_ptr());

    for (int64_t row = 0; row < rows; ++row) {
        char* dst_row = cpu_kv_base
            + static_cast<size_t>(row_start + row) * cpu_kv_row_pitch_bytes
            + static_cast<size_t>(token_start * 2 * dim);
        const char* src_key = key_base + static_cast<size_t>(row) * gpu_code_row_pitch_bytes;
        const char* src_value = value_base + static_cast<size_t>(row) * gpu_code_row_pitch_bytes;

        C10_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_row,
            code_dst_pitch_bytes,
            src_key,
            code_src_pitch_bytes,
            code_width_bytes,
            static_cast<size_t>(tokens),
            cudaMemcpyDeviceToHost,
            stream
        ));
        C10_CUDA_CHECK(cudaMemcpy2DAsync(
            dst_row + code_width_bytes,
            code_dst_pitch_bytes,
            src_value,
            code_src_pitch_bytes,
            code_width_bytes,
            static_cast<size_t>(tokens),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }

    const size_t scale_elem_bytes = sizeof(float);
    const size_t scale_width_bytes = static_cast<size_t>(tokens) * scale_elem_bytes;
    const size_t scale_src_pitch_bytes = scale_width_bytes;
    const size_t scale_dst_pitch_bytes = static_cast<size_t>(total_tokens) * scale_elem_bytes;
    const char* scale_src = static_cast<const char*>(value_scale.data_ptr());
    char* scale_dst = static_cast<char*>(cpu_value_scale.data_ptr())
        + (static_cast<size_t>(row_start) * static_cast<size_t>(total_tokens)
           + static_cast<size_t>(token_start)) * scale_elem_bytes;

    C10_CUDA_CHECK(cudaMemcpy2DAsync(
        scale_dst,
        scale_dst_pitch_bytes,
        scale_src,
        scale_src_pitch_bytes,
        scale_width_bytes,
        static_cast<size_t>(rows),
        cudaMemcpyDeviceToHost,
        stream
    ));
}



void concat_static_recent_lookup_gather_uva_kv_update_cache(
    torch::Tensor static_keys,
    torch::Tensor static_values,
    torch::Tensor recent_keys,
    torch::Tensor recent_values,
    torch::Tensor request_token_ids,
    torch::Tensor cpu_kv,
    torch::Tensor out_keys,
    torch::Tensor out_values,
    torch::Tensor hit_mask,
    torch::Tensor cache_token_ids,
    torch::Tensor cache_locks,
    torch::Tensor cache_keys,
    torch::Tensor cache_values,
    int64_t static_len,
    int64_t recent_len,
    int64_t sparse_len,
    c10::optional<torch::Tensor> cache_stamps = c10::nullopt,
    c10::optional<torch::Tensor> step_counter = c10::nullopt,
    int64_t ways = 2,
    c10::optional<torch::Tensor> prio_buckets = c10::nullopt,
    c10::optional<torch::Tensor> prio_scores = c10::nullopt,
    c10::optional<torch::Tensor> cache_prio = c10::nullopt) {
    check_cuda_tensor(static_keys, "static_keys");
    check_cuda_tensor(static_values, "static_values");
    check_cuda_tensor(recent_keys, "recent_keys");
    check_cuda_tensor(recent_values, "recent_values");
    check_cuda_tensor(request_token_ids, "request_token_ids");
    check_cpu_tensor(cpu_kv, "cpu_kv");
    check_cuda_tensor(out_keys, "out_keys");
    check_cuda_tensor(out_values, "out_values");
    check_cuda_tensor(hit_mask, "hit_mask");
    check_cuda_tensor(cache_token_ids, "cache_token_ids");
    check_cuda_tensor(cache_locks, "cache_locks");
    check_cuda_tensor(cache_keys, "cache_keys");
    check_cuda_tensor(cache_values, "cache_values");
    check_two_byte_dtype(static_keys, "static_keys");
    check_two_byte_dtype(static_values, "static_values");
    check_two_byte_dtype(recent_keys, "recent_keys");
    check_two_byte_dtype(recent_values, "recent_values");
    check_two_byte_dtype(cpu_kv, "cpu_kv");
    check_two_byte_dtype(out_keys, "out_keys");
    check_two_byte_dtype(out_values, "out_values");
    check_two_byte_dtype(cache_keys, "cache_keys");
    check_two_byte_dtype(cache_values, "cache_values");

    TORCH_CHECK(request_token_ids.scalar_type() == torch::kInt32, "request_token_ids must be int32");
    TORCH_CHECK(hit_mask.scalar_type() == torch::kInt32, "hit_mask must be int32");
    TORCH_CHECK(cache_token_ids.scalar_type() == torch::kInt32, "cache_token_ids must be int32");
    TORCH_CHECK(cache_locks.scalar_type() == torch::kInt32, "cache_locks must be int32");
    TORCH_CHECK(static_keys.scalar_type() == static_values.scalar_type(), "static key/value dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == recent_keys.scalar_type(), "recent key dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == recent_values.scalar_type(), "recent value dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == out_keys.scalar_type(), "output key dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == out_values.scalar_type(), "output value dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == cache_keys.scalar_type(), "cache key dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == cache_values.scalar_type(), "cache value dtype mismatch");
    TORCH_CHECK(static_keys.dim() == 3 && static_values.dim() == 3, "static tensors must be [rows, len, dim]");
    TORCH_CHECK(recent_keys.dim() == 3 && recent_values.dim() == 3, "recent tensors must be [rows, len, dim]");
    TORCH_CHECK(request_token_ids.dim() == 2, "request_token_ids must be [rows, request_count]");
    TORCH_CHECK(cpu_kv.dim() == 4, "cpu_kv must be [rows, total_tokens, 2, dim]");
    TORCH_CHECK(out_keys.dim() == 4 && out_values.dim() == 4, "output tensors must be [rows, len, 1, dim]");
    TORCH_CHECK(hit_mask.dim() == 2, "hit_mask must be [rows, request_count]");
    TORCH_CHECK(cache_token_ids.dim() == 2, "cache_token_ids must be [rows, cache_size]");
    TORCH_CHECK(cache_locks.dim() == 2, "cache_locks must be [rows, cache_size]");
    TORCH_CHECK(cache_keys.dim() == 3 && cache_values.dim() == 3, "cache tensors must be [rows, cache_size, dim]");
    TORCH_CHECK(static_len >= 0 && recent_len >= 0 && sparse_len >= 0, "concat lengths must be non-negative");

    const int rows = static_cast<int>(static_keys.size(0));
    const int static_capacity = static_cast<int>(static_keys.size(1));
    const int recent_capacity = static_cast<int>(recent_keys.size(1));
    const int dim = static_cast<int>(static_keys.size(2));
    const int request_count = static_cast<int>(request_token_ids.size(1));
    const int total_tokens = static_cast<int>(cpu_kv.size(1));
    const int cache_size = static_cast<int>(cache_token_ids.size(1));
    const int total_len = static_cast<int>(static_len + recent_len + sparse_len);

    TORCH_CHECK(dim > 0, "dim must be positive");
    TORCH_CHECK(static_len <= static_capacity, "static_len exceeds static capacity");
    TORCH_CHECK(recent_len <= recent_capacity, "recent_len exceeds recent capacity");
    TORCH_CHECK(sparse_len <= request_count, "sparse_len exceeds request_count");
    TORCH_CHECK(static_values.size(0) == rows && static_values.size(1) == static_capacity && static_values.size(2) == dim,
                "static_values shape mismatch");
    TORCH_CHECK(recent_keys.size(0) == rows && recent_values.size(0) == rows, "recent row mismatch");
    TORCH_CHECK(recent_values.size(1) == recent_capacity && recent_keys.size(2) == dim && recent_values.size(2) == dim,
                "recent shape mismatch");
    TORCH_CHECK(request_token_ids.size(0) == rows, "request row mismatch");
    TORCH_CHECK(cpu_kv.size(0) == rows && cpu_kv.size(2) == 2 && cpu_kv.size(3) == dim, "cpu_kv shape mismatch");
    TORCH_CHECK(out_keys.size(0) == rows && out_values.size(0) == rows, "output row mismatch");
    TORCH_CHECK(out_keys.size(1) >= total_len && out_values.size(1) == out_keys.size(1),
                "output rows must be at least total_len wide (out_row_len >= total_len)");
    const int out_row_len = static_cast<int>(out_keys.size(1));
    TORCH_CHECK(out_keys.size(2) == 1 && out_values.size(2) == 1, "output head dimension must be 1");
    TORCH_CHECK(out_keys.size(3) == dim && out_values.size(3) == dim, "output dim mismatch");
    TORCH_CHECK(hit_mask.size(0) == rows && hit_mask.size(1) == request_count, "hit_mask shape mismatch");
    TORCH_CHECK(cache_locks.size(0) == rows && cache_locks.size(1) == cache_size, "cache_locks shape mismatch");
    TORCH_CHECK(cache_keys.size(0) == rows && cache_values.size(0) == rows, "cache row mismatch");
    TORCH_CHECK(cache_keys.size(1) == cache_size && cache_values.size(1) == cache_size, "cache_size mismatch");
    TORCH_CHECK(cache_keys.size(2) == dim && cache_values.size(2) == dim, "cache dim mismatch");

    // Optional W-way LRU cache: pass a per-slot stamp buffer + device step counter to switch the
    // kernel from legacy 2-way always-insert to W-way LRU replacement. Absent -> bit-identical old.
    int32_t* cache_stamps_ptr = nullptr;
    const int32_t* step_ptr = nullptr;
    int cache_ways = 2;
    if (cache_stamps.has_value()) {
        const torch::Tensor& st = cache_stamps.value();
        check_cuda_tensor(st, "cache_stamps");
        TORCH_CHECK(st.scalar_type() == torch::kInt32, "cache_stamps must be int32");
        TORCH_CHECK(st.sizes().vec() == std::vector<int64_t>({rows, cache_size}),
                    "cache_stamps must be [rows, cache_size]");
        TORCH_CHECK(ways >= 2 && ways <= 16 && cache_size / ways >= 1,
                    "ways must be in [2, 16] and cache_size / ways >= 1");
        cache_stamps_ptr = st.data_ptr<int32_t>();
        cache_ways = static_cast<int>(ways);
        if (step_counter.has_value()) {
            const torch::Tensor& sc = step_counter.value();
            check_cuda_tensor(sc, "step_counter");
            TORCH_CHECK(sc.scalar_type() == torch::kInt32, "step_counter must be int32");
            TORCH_CHECK(sc.numel() == 1, "step_counter must be a 1-element int32 tensor");
            step_ptr = sc.data_ptr<int32_t>();
        }
    }

    // Optional "score" replacement policy: a per-(row, token) priority buffer written by the
    // selector this step — int16 priority bucket codes (smaller = better) OR fp32 selector
    // scores (larger = better) — switches the LRU victim choice to worst-current-priority.
    // Requires the stamp/step LRU state (ways > 2); its width may exceed cpu_kv's token count
    // (signature capacity is rounded up), so the kernel gets the stride explicitly.
    const int16_t* prio_buckets_ptr = nullptr;
    const float* prio_scores_ptr = nullptr;
    int prio_stride = 0;
    if (prio_buckets.has_value() || prio_scores.has_value()) {
        TORCH_CHECK(!(prio_buckets.has_value() && prio_scores.has_value()),
                    "prio_buckets and prio_scores are mutually exclusive");
        TORCH_CHECK(cache_stamps_ptr != nullptr && step_ptr != nullptr,
                    "score policy requires cache_stamps and step_counter (W-way LRU state)");
        const torch::Tensor& prio = prio_buckets.has_value() ? prio_buckets.value() : prio_scores.value();
        check_cuda_tensor(prio, "prio");
        TORCH_CHECK(prio.dim() == 2 && prio.size(0) == rows && prio.size(1) >= total_tokens,
                    "prio must be [rows, >= total_tokens]");
        TORCH_CHECK(prio.is_contiguous(), "prio must be contiguous");
        prio_stride = static_cast<int>(prio.size(1));
        if (prio_buckets.has_value()) {
            TORCH_CHECK(prio.scalar_type() == torch::kInt16, "prio_buckets must be int16");
            prio_buckets_ptr = prio.data_ptr<int16_t>();
        } else {
            TORCH_CHECK(prio.scalar_type() == torch::kFloat32, "prio_scores must be float32");
            prio_scores_ptr = prio.data_ptr<float>();
        }
    }
    int16_t* cache_prio_ptr = nullptr;
    if (cache_prio.has_value()) {
        const torch::Tensor& cp = cache_prio.value();
        check_cuda_tensor(cp, "cache_prio");
        TORCH_CHECK(prio_buckets_ptr != nullptr, "cache_prio requires prio_buckets (bucket mode)");
        TORCH_CHECK(cp.scalar_type() == torch::kInt16, "cache_prio must be int16");
        TORCH_CHECK(cp.sizes().vec() == std::vector<int64_t>({rows, cache_size}),
                    "cache_prio must be [rows, cache_size]");
        cache_prio_ptr = cp.data_ptr<int16_t>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(rows, total_len);
    cometkv::concat_static_recent_lookup_gather_uva_kv_update_cache_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(static_keys.data_ptr()),
        reinterpret_cast<const uint16_t*>(static_values.data_ptr()),
        reinterpret_cast<const uint16_t*>(recent_keys.data_ptr()),
        reinterpret_cast<const uint16_t*>(recent_values.data_ptr()),
        request_token_ids.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(cpu_kv.data_ptr()),
        reinterpret_cast<uint16_t*>(out_keys.data_ptr()),
        reinterpret_cast<uint16_t*>(out_values.data_ptr()),
        hit_mask.data_ptr<int32_t>(),
        cache_token_ids.data_ptr<int32_t>(),
        cache_locks.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(cache_keys.data_ptr()),
        reinterpret_cast<uint16_t*>(cache_values.data_ptr()),
        rows,
        static_capacity,
        recent_capacity,
        request_count,
        total_tokens,
        cache_size,
        static_cast<int>(static_len),
        static_cast<int>(recent_len),
        static_cast<int>(sparse_len),
        total_len,
        out_row_len,
        dim,
        cache_stamps_ptr,
        step_ptr,
        cache_ways,
        prio_buckets_ptr,
        prio_scores_ptr,
        prio_stride,
        cache_prio_ptr
    );
}


void concat_static_recent_lookup_gather_uva_kv_update_cache_int8(
    torch::Tensor static_keys,
    torch::Tensor static_values,
    torch::Tensor recent_keys,
    torch::Tensor recent_values,
    torch::Tensor request_token_ids,
    torch::Tensor cpu_kv_int8,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor out_keys,
    torch::Tensor out_values,
    torch::Tensor hit_mask,
    torch::Tensor cache_token_ids,
    torch::Tensor cache_locks,
    torch::Tensor cache_keys,
    torch::Tensor cache_values,
    int64_t static_len,
    int64_t recent_len,
    int64_t sparse_len) {
    check_cuda_tensor(static_keys, "static_keys");
    check_cuda_tensor(static_values, "static_values");
    check_cuda_tensor(recent_keys, "recent_keys");
    check_cuda_tensor(recent_values, "recent_values");
    check_cuda_tensor(request_token_ids, "request_token_ids");
    check_cpu_tensor(cpu_kv_int8, "cpu_kv_int8");
    check_cuda_tensor(k_scale, "k_scale");
    check_cpu_tensor(v_scale, "v_scale");
    check_cuda_tensor(out_keys, "out_keys");
    check_cuda_tensor(out_values, "out_values");
    check_cuda_tensor(hit_mask, "hit_mask");
    check_cuda_tensor(cache_token_ids, "cache_token_ids");
    check_cuda_tensor(cache_locks, "cache_locks");
    check_cuda_tensor(cache_keys, "cache_keys");
    check_cuda_tensor(cache_values, "cache_values");
    check_two_byte_dtype(static_keys, "static_keys");
    check_two_byte_dtype(static_values, "static_values");
    check_two_byte_dtype(recent_keys, "recent_keys");
    check_two_byte_dtype(recent_values, "recent_values");
    check_two_byte_dtype(out_keys, "out_keys");
    check_two_byte_dtype(out_values, "out_values");
    check_two_byte_dtype(cache_keys, "cache_keys");
    check_two_byte_dtype(cache_values, "cache_values");

    TORCH_CHECK(cpu_kv_int8.scalar_type() == torch::kChar, "cpu_kv_int8 must be int8");
    TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32, "k_scale must be float32");
    TORCH_CHECK(v_scale.scalar_type() == torch::kFloat32, "v_scale must be float32");
    TORCH_CHECK(v_scale.is_pinned(), "v_scale must be pinned for UVA access");
    TORCH_CHECK(request_token_ids.scalar_type() == torch::kInt32, "request_token_ids must be int32");
    TORCH_CHECK(hit_mask.scalar_type() == torch::kInt32, "hit_mask must be int32");
    TORCH_CHECK(cache_token_ids.scalar_type() == torch::kInt32, "cache_token_ids must be int32");
    TORCH_CHECK(cache_locks.scalar_type() == torch::kInt32, "cache_locks must be int32");
    TORCH_CHECK(static_keys.scalar_type() == out_keys.scalar_type(), "static/out key dtype mismatch");
    TORCH_CHECK(static_keys.scalar_type() == cache_keys.scalar_type(), "static/cache key dtype mismatch");
    TORCH_CHECK(static_keys.dim() == 3 && static_values.dim() == 3, "static tensors must be [rows, len, dim]");
    TORCH_CHECK(recent_keys.dim() == 3 && recent_values.dim() == 3, "recent tensors must be [rows, len, dim]");
    TORCH_CHECK(request_token_ids.dim() == 2, "request_token_ids must be [rows, request_count]");
    TORCH_CHECK(cpu_kv_int8.dim() == 4, "cpu_kv_int8 must be [rows, total_tokens, 2, dim]");
    TORCH_CHECK(out_keys.dim() == 4 && out_values.dim() == 4, "output tensors must be [rows, len, 1, dim]");

    const int rows = static_cast<int>(static_keys.size(0));
    const int static_capacity = static_cast<int>(static_keys.size(1));
    const int recent_capacity = static_cast<int>(recent_keys.size(1));
    const int dim = static_cast<int>(static_keys.size(2));
    const int request_count = static_cast<int>(request_token_ids.size(1));
    const int total_tokens = static_cast<int>(cpu_kv_int8.size(1));
    const int cache_size = static_cast<int>(cache_token_ids.size(1));
    const int total_len = static_cast<int>(static_len + recent_len + sparse_len);

    TORCH_CHECK(dim > 0, "dim must be positive");
    TORCH_CHECK(static_len <= static_capacity, "static_len exceeds static capacity");
    TORCH_CHECK(recent_len <= recent_capacity, "recent_len exceeds recent capacity");
    TORCH_CHECK(sparse_len <= request_count, "sparse_len exceeds request_count");
    TORCH_CHECK(cpu_kv_int8.size(0) == rows && cpu_kv_int8.size(2) == 2 && cpu_kv_int8.size(3) == dim,
                "cpu_kv_int8 shape mismatch");
    TORCH_CHECK(k_scale.dim() == 2 && k_scale.size(0) == rows && k_scale.size(1) == dim,
                "k_scale must be [rows, dim]");
    TORCH_CHECK(v_scale.dim() == 2 && v_scale.size(0) == rows && v_scale.size(1) == total_tokens,
                "v_scale must be [rows, total_tokens]");
    TORCH_CHECK(out_keys.size(0) == rows && out_keys.size(1) >= total_len && out_keys.size(2) == 1 &&
                out_keys.size(3) == dim, "out_keys shape mismatch");
    TORCH_CHECK(out_values.size(1) == out_keys.size(1), "out_values width mismatch");
    const int out_row_len = static_cast<int>(out_keys.size(1));
    TORCH_CHECK(hit_mask.size(0) == rows && hit_mask.size(1) == request_count, "hit_mask shape mismatch");
    TORCH_CHECK(cache_keys.size(0) == rows && cache_keys.size(1) == cache_size && cache_keys.size(2) == dim,
                "cache_keys shape mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(rows, total_len);
    if (out_keys.scalar_type() == torch::kFloat16) {
        cometkv::concat_static_recent_lookup_gather_uva_kv_update_cache_int8_kernel<half><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const half*>(static_keys.data_ptr()),
            reinterpret_cast<const half*>(static_values.data_ptr()),
            reinterpret_cast<const half*>(recent_keys.data_ptr()),
            reinterpret_cast<const half*>(recent_values.data_ptr()),
            request_token_ids.data_ptr<int32_t>(),
            cpu_kv_int8.data_ptr<int8_t>(),
            k_scale.data_ptr<float>(),
            v_scale.data_ptr<float>(),
            reinterpret_cast<half*>(out_keys.data_ptr()),
            reinterpret_cast<half*>(out_values.data_ptr()),
            hit_mask.data_ptr<int32_t>(),
            cache_token_ids.data_ptr<int32_t>(),
            cache_locks.data_ptr<int32_t>(),
            reinterpret_cast<half*>(cache_keys.data_ptr()),
            reinterpret_cast<half*>(cache_values.data_ptr()),
            rows, static_capacity, recent_capacity, request_count, total_tokens, cache_size,
            static_cast<int>(static_len), static_cast<int>(recent_len), static_cast<int>(sparse_len),
            total_len, out_row_len, dim);
    } else if (out_keys.scalar_type() == torch::kBFloat16) {
        cometkv::concat_static_recent_lookup_gather_uva_kv_update_cache_int8_kernel<nv_bfloat16><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(static_keys.data_ptr()),
            reinterpret_cast<const nv_bfloat16*>(static_values.data_ptr()),
            reinterpret_cast<const nv_bfloat16*>(recent_keys.data_ptr()),
            reinterpret_cast<const nv_bfloat16*>(recent_values.data_ptr()),
            request_token_ids.data_ptr<int32_t>(),
            cpu_kv_int8.data_ptr<int8_t>(),
            k_scale.data_ptr<float>(),
            v_scale.data_ptr<float>(),
            reinterpret_cast<nv_bfloat16*>(out_keys.data_ptr()),
            reinterpret_cast<nv_bfloat16*>(out_values.data_ptr()),
            hit_mask.data_ptr<int32_t>(),
            cache_token_ids.data_ptr<int32_t>(),
            cache_locks.data_ptr<int32_t>(),
            reinterpret_cast<nv_bfloat16*>(cache_keys.data_ptr()),
            reinterpret_cast<nv_bfloat16*>(cache_values.data_ptr()),
            rows, static_capacity, recent_capacity, request_count, total_tokens, cache_size,
            static_cast<int>(static_len), static_cast<int>(recent_len), static_cast<int>(sparse_len),
            total_len, out_row_len, dim);
    } else {
        TORCH_CHECK(false, "out_keys must be fp16 or bf16");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("append_lockstep_local_kv_cache", &append_lockstep_local_kv_cache,
          "Append one decode KV token to lockstep fixed local window without token ids (CUDA)",
          py::arg("key_states"), py::arg("value_states"), py::arg("local_keys"), py::arg("local_values"),
          py::arg("decode_step"), py::arg("local_capacity"), py::arg("slide_stride"));
    m.def("append_lockstep_local_kv_cache_and_advance", &append_lockstep_local_kv_cache_and_advance,
          "Append one decode KV token to lockstep fixed local window and advance visible lengths (CUDA)",
          py::arg("key_states"), py::arg("value_states"), py::arg("local_keys"), py::arg("local_values"),
          py::arg("visible_lengths"), py::arg("decode_step"), py::arg("local_capacity"), py::arg("slide_stride"));
    m.def("append_lockstep_local_kv_cache_dev", &append_lockstep_local_kv_cache_dev,
          "CUDA-graph capturable append (no advance) reading decode_step from a device int32 counter (CUDA)",
          py::arg("key_states"), py::arg("value_states"), py::arg("local_keys"), py::arg("local_values"),
          py::arg("decode_step_dev"), py::arg("local_capacity"), py::arg("slide_stride"));
    m.def("refresh_static_prompt_recent_state", &refresh_static_prompt_recent_state,
          "Refresh CometKV static prompt recent suffix and static lengths (CUDA)",
          py::arg("prompt_lengths"), py::arg("visible_lengths"), py::arg("prompt_recent_keys"), py::arg("prompt_recent_values"),
          py::arg("static_keys"), py::arg("static_values"), py::arg("static_lengths"),
          py::arg("static_pattern_start"), py::arg("static_pattern_end"));
    m.def("copy_interleaved_kv_to_cpu", &copy_interleaved_kv_to_cpu,
          "Copy contiguous GPU K/V rows into interleaved pinned CPU KV storage with cudaMemcpy2DAsync",
          py::arg("key_rows"), py::arg("value_rows"), py::arg("cpu_kv"),
          py::arg("row_start"), py::arg("token_start"));
    m.def("copy_interleaved_int8_kv_to_cpu", &copy_interleaved_int8_kv_to_cpu,
          "Copy contiguous GPU int8 K/V rows and V scales into pinned CPU storage with cudaMemcpy2DAsync",
          py::arg("key_codes"), py::arg("value_codes"), py::arg("value_scale"),
          py::arg("cpu_kv_int8"), py::arg("cpu_value_scale"),
          py::arg("row_start"), py::arg("token_start"));
    m.def("concat_static_recent_lookup_gather_uva_kv_update_cache", &concat_static_recent_lookup_gather_uva_kv_update_cache,
          "Concat static/recent KV and lookup-gather sparse KV directly into a FlashAttention buffer (CUDA)",
          py::arg("static_keys"), py::arg("static_values"), py::arg("recent_keys"), py::arg("recent_values"),
          py::arg("request_token_ids"), py::arg("cpu_kv"), py::arg("out_keys"), py::arg("out_values"),
          py::arg("hit_mask"), py::arg("cache_token_ids"), py::arg("cache_locks"), py::arg("cache_keys"),
          py::arg("cache_values"), py::arg("static_len"), py::arg("recent_len"), py::arg("sparse_len"),
          py::arg("cache_stamps") = c10::nullopt, py::arg("step_counter") = c10::nullopt,
          py::arg("ways") = 2,
          py::arg("prio_buckets") = c10::nullopt, py::arg("prio_scores") = c10::nullopt,
          py::arg("cache_prio") = c10::nullopt);
    m.def("concat_static_recent_lookup_gather_uva_kv_update_cache_int8",
          &concat_static_recent_lookup_gather_uva_kv_update_cache_int8,
          "int8 cpu_kv variant of the concat gather: dequant K per-channel / V per-token to bf16/fp16 (CUDA)",
          py::arg("static_keys"), py::arg("static_values"), py::arg("recent_keys"), py::arg("recent_values"),
          py::arg("request_token_ids"), py::arg("cpu_kv_int8"), py::arg("k_scale"), py::arg("v_scale"),
          py::arg("out_keys"), py::arg("out_values"), py::arg("hit_mask"), py::arg("cache_token_ids"),
          py::arg("cache_locks"), py::arg("cache_keys"), py::arg("cache_values"),
          py::arg("static_len"), py::arg("recent_len"), py::arg("sparse_len"));
}
