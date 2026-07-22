#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Dedicated GPU-resident gather kernels for the CometKV_GPU backend.
//
// These mirror concat_static_recent_lookup_gather_uva_kv_update_cache[_int8] in
// cometkv_gather_kernel.cuh, but the retrieval store (`gpu_kv`) lives in device
// memory, so there is no PCIe/UVA traffic and no GPU token-cache to amortise it.
// Every token-cache concept (slots, locks, atomics, hit_mask, cache write-back)
// is dropped: a sparse token is read straight from `gpu_kv[row, token, ...]`.
// Kept fully isolated from the UVA kernels so each can be tuned independently.

namespace cometkv_gpu {

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to(float v);
template <>
__device__ __forceinline__ half float_to<half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ nv_bfloat16 float_to<nv_bfloat16>(float v) { return __float2bfloat16(v); }


// bf16/fp16 (2-byte) gather: copy [static | recent] then gather sparse tokens
// directly from the device-resident gpu_kv store.
__global__ void concat_static_recent_gpu_gather_kernel(
    const uint16_t* static_keys,
    const uint16_t* static_values,
    const uint16_t* recent_keys,
    const uint16_t* recent_values,
    const int32_t* request_token_ids,
    const uint16_t* gpu_kv,
    uint16_t* out_keys,
    uint16_t* out_values,
    int rows,
    int static_capacity,
    int recent_capacity,
    int request_count,
    int total_tokens,
    int static_len,
    int recent_len,
    int sparse_len,
    int total_len,
    int out_row_len,
    int dim) {
    const int row = blockIdx.x;
    const int token_idx = blockIdx.y;
    const int lane = threadIdx.x;

    if (row >= rows || token_idx >= total_len) {
        return;
    }

    if (dim == 128) {
        constexpr int vec_count = 16;
        const uint4* static_key_vec = reinterpret_cast<const uint4*>(static_keys);
        const uint4* static_value_vec = reinterpret_cast<const uint4*>(static_values);
        const uint4* recent_key_vec = reinterpret_cast<const uint4*>(recent_keys);
        const uint4* recent_value_vec = reinterpret_cast<const uint4*>(recent_values);
        const uint4* gpu_vec = reinterpret_cast<const uint4*>(gpu_kv);
        uint4* out_key_vec = reinterpret_cast<uint4*>(out_keys);
        uint4* out_value_vec = reinterpret_cast<uint4*>(out_values);
        const int out_vec_offset = (row * out_row_len + token_idx) * vec_count;

        if (token_idx < static_len) {
            const int src_vec_offset = (row * static_capacity + token_idx) * vec_count;
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = static_key_vec[src_vec_offset + lane];
                out_value_vec[out_vec_offset + lane] = static_value_vec[src_vec_offset + lane];
            }
            return;
        }

        if (token_idx < static_len + recent_len) {
            const int recent_idx = token_idx - static_len;
            const int src_vec_offset = (row * recent_capacity + recent_idx) * vec_count;
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = recent_key_vec[src_vec_offset + lane];
                out_value_vec[out_vec_offset + lane] = recent_value_vec[src_vec_offset + lane];
            }
            return;
        }

        const int request_idx = token_idx - static_len - recent_len;
        if (request_idx >= sparse_len || request_idx >= request_count) {
            return;
        }
        const int token = request_token_ids[row * request_count + request_idx];
        if (token < 0) {
            const uint4 zero = make_uint4(0, 0, 0, 0);
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = zero;
                out_value_vec[out_vec_offset + lane] = zero;
            }
            return;
        }

        const int src_vec_offset = ((row * total_tokens + token) * 2) * vec_count;
        if (lane < vec_count) {
            out_key_vec[out_vec_offset + lane] = gpu_vec[src_vec_offset + lane];
            out_value_vec[out_vec_offset + lane] = gpu_vec[src_vec_offset + vec_count + lane];
        }
        return;
    }

    // Generic-dim scalar fallback.
    const int out_offset = (row * out_row_len + token_idx) * dim;
    if (token_idx < static_len) {
        const int src_offset = (row * static_capacity + token_idx) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = static_keys[src_offset + col];
            out_values[out_offset + col] = static_values[src_offset + col];
        }
        return;
    }

    if (token_idx < static_len + recent_len) {
        const int recent_idx = token_idx - static_len;
        const int src_offset = (row * recent_capacity + recent_idx) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = recent_keys[src_offset + col];
            out_values[out_offset + col] = recent_values[src_offset + col];
        }
        return;
    }

    const int request_idx = token_idx - static_len - recent_len;
    if (request_idx >= sparse_len || request_idx >= request_count) {
        return;
    }
    const int token = request_token_ids[row * request_count + request_idx];
    if (token < 0) {
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = 0;
            out_values[out_offset + col] = 0;
        }
        return;
    }

    const int src_offset = ((row * total_tokens + token) * 2) * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        out_keys[out_offset + col] = gpu_kv[src_offset + col];
        out_values[out_offset + col] = gpu_kv[src_offset + dim + col];
    }
}


// int8 variant: gpu_kv is int8; K dequantized per-channel (k_scale[row, channel]),
// V per-token (v_scale[row, token]). static/recent stay scalar_t.
template <typename scalar_t>
__global__ void concat_static_recent_gpu_gather_int8_kernel(
    const scalar_t* static_keys,
    const scalar_t* static_values,
    const scalar_t* recent_keys,
    const scalar_t* recent_values,
    const int32_t* request_token_ids,
    const int8_t* gpu_kv_int8,
    const float* k_scale,
    const float* v_scale,
    scalar_t* out_keys,
    scalar_t* out_values,
    int rows,
    int static_capacity,
    int recent_capacity,
    int request_count,
    int total_tokens,
    int static_len,
    int recent_len,
    int sparse_len,
    int total_len,
    int out_row_len,
    int dim) {
    const int row = blockIdx.x;
    const int token_idx = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || token_idx >= total_len) {
        return;
    }

    if (dim == 128) {
        constexpr int vec_count = 16;   // 16 lanes * 8 scalar_t = 128
        const uint4* static_key_vec = reinterpret_cast<const uint4*>(static_keys);
        const uint4* static_value_vec = reinterpret_cast<const uint4*>(static_values);
        const uint4* recent_key_vec = reinterpret_cast<const uint4*>(recent_keys);
        const uint4* recent_value_vec = reinterpret_cast<const uint4*>(recent_values);
        uint4* out_key_vec = reinterpret_cast<uint4*>(out_keys);
        uint4* out_value_vec = reinterpret_cast<uint4*>(out_values);
        const int out_vec_offset = (row * out_row_len + token_idx) * vec_count;

        if (token_idx < static_len) {
            const int src = (row * static_capacity + token_idx) * vec_count;
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = static_key_vec[src + lane];
                out_value_vec[out_vec_offset + lane] = static_value_vec[src + lane];
            }
            return;
        }
        if (token_idx < static_len + recent_len) {
            const int ridx = token_idx - static_len;
            const int src = (row * recent_capacity + ridx) * vec_count;
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = recent_key_vec[src + lane];
                out_value_vec[out_vec_offset + lane] = recent_value_vec[src + lane];
            }
            return;
        }

        const int request_idx = token_idx - static_len - recent_len;
        if (request_idx >= sparse_len || request_idx >= request_count) {
            return;
        }
        const int token = request_token_ids[row * request_count + request_idx];
        if (token < 0) {
            const uint4 zero = make_uint4(0, 0, 0, 0);
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = zero;
                out_value_vec[out_vec_offset + lane] = zero;
            }
            return;
        }

        if (lane < vec_count) {
            const int c0 = lane * 8;
            const int k_code_base = ((row * total_tokens + token) * 2 + 0) * dim + c0;
            const int v_code_base = ((row * total_tokens + token) * 2 + 1) * dim + c0;
            const int k_scale_base = row * dim + c0;
            const float v_token_scale = v_scale[row * total_tokens + token];
            union { scalar_t s[8]; uint4 v; } kpack, vpack;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                kpack.s[i] = float_to<scalar_t>(static_cast<float>(gpu_kv_int8[k_code_base + i]) * k_scale[k_scale_base + i]);
                vpack.s[i] = float_to<scalar_t>(static_cast<float>(gpu_kv_int8[v_code_base + i]) * v_token_scale);
            }
            out_key_vec[out_vec_offset + lane] = kpack.v;
            out_value_vec[out_vec_offset + lane] = vpack.v;
        }
        return;
    }

    // Generic-dim scalar fallback.
    const int out_offset = (row * out_row_len + token_idx) * dim;
    if (token_idx < static_len) {
        const int src_offset = (row * static_capacity + token_idx) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = static_keys[src_offset + col];
            out_values[out_offset + col] = static_values[src_offset + col];
        }
        return;
    }
    if (token_idx < static_len + recent_len) {
        const int recent_idx = token_idx - static_len;
        const int src_offset = (row * recent_capacity + recent_idx) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = recent_keys[src_offset + col];
            out_values[out_offset + col] = recent_values[src_offset + col];
        }
        return;
    }

    const int request_idx = token_idx - static_len - recent_len;
    if (request_idx >= sparse_len || request_idx >= request_count) {
        return;
    }
    const int token = request_token_ids[row * request_count + request_idx];
    if (token < 0) {
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = float_to<scalar_t>(0.0f);
            out_values[out_offset + col] = float_to<scalar_t>(0.0f);
        }
        return;
    }

    const int k_code_base = ((row * total_tokens + token) * 2 + 0) * dim;
    const int v_code_base = ((row * total_tokens + token) * 2 + 1) * dim;
    const float v_token_scale = v_scale[row * total_tokens + token];
    const int k_scale_base = row * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        const float k_val = static_cast<float>(gpu_kv_int8[k_code_base + col]) * k_scale[k_scale_base + col];
        const float v_val = static_cast<float>(gpu_kv_int8[v_code_base + col]) * v_token_scale;
        out_keys[out_offset + col] = float_to<scalar_t>(k_val);
        out_values[out_offset + col] = float_to<scalar_t>(v_val);
    }
}

}  // namespace cometkv_gpu
