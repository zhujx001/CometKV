#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cometkv {

constexpr int kMaxHammingDistance = 128;
constexpr int kMaxRanges = 2;
constexpr int kHammingTopkThreads = 256;
constexpr int kHammingTopkChunkSize = kHammingTopkThreads;

__device__ __forceinline__ float scalar_to_float(const half value) {
    return __half2float(value);
}

__device__ __forceinline__ float scalar_to_float(const nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ int positive_range_length(int start, int end) {
    return end > start ? (end - start) : 0;
}

__device__ __forceinline__ int topk_total_candidates_for_row(
    const int32_t* range_starts,
    const int32_t* range_ends,
    int row) {
    int total = 0;
    #pragma unroll
    for (int range_idx = 0; range_idx < kMaxRanges; ++range_idx) {
        const int start = range_starts[row * kMaxRanges + range_idx];
        const int end = range_ends[row * kMaxRanges + range_idx];
        total += positive_range_length(start, end);
    }
    return total;
}

__device__ __forceinline__ int descending_rank_to_token(
    const int32_t* range_starts,
    const int32_t* range_ends,
    int row,
    int rank) {
    // CometKV decode only has two ranges and range 1 always contains newer tokens than range 0.
    const int newer_start = range_starts[row * kMaxRanges + 1];
    const int newer_end = range_ends[row * kMaxRanges + 1];
    const int newer_len = positive_range_length(newer_start, newer_end);
    if (rank < newer_len) {
        return newer_end - 1 - rank;
    }

    const int older_start = range_starts[row * kMaxRanges + 0];
    const int older_end = range_ends[row * kMaxRanges + 0];
    return older_end - 1 - (rank - newer_len);
}

__global__ void asym_signature_score_kernel(
    const float* query_proj,        // [rows, proj_width] fp32, x = P . (sum of group query heads)
    const float* query_proj_total,  // [rows] fp32, sum of the first sig_bits_used entries of x
    const uint8_t* keys,            // [rows, tokens, sig_bytes]
    const int32_t* range_starts,    // [rows, kMaxRanges]
    const int32_t* range_ends,
    const float* norm_lo,           // [rows] or nullptr
    const float* norm_step,         // [rows] or nullptr
    float* scores,                  // [rows, tokens]
    int rows,
    int tokens,
    int sig_bytes,
    int sig_bits_used,
    int proj_width,
    int max_chunks) {
    const int row = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;
    if (row >= rows || chunk_idx >= max_chunks) {
        return;
    }

    __shared__ float shared_x[kMaxHammingDistance];  // sig_bits_used <= 128
    for (int i = thread_idx; i < sig_bits_used; i += blockDim.x) {
        shared_x[i] = query_proj[row * proj_width + i];
    }
    __syncthreads();

    const int total_candidates = topk_total_candidates_for_row(range_starts, range_ends, row);
    const int candidate_rank = chunk_idx * kHammingTopkChunkSize + thread_idx;
    if (candidate_rank >= total_candidates) {
        return;
    }
    const int token = descending_rank_to_token(range_starts, range_ends, row, candidate_rank);
    const uint8_t* sig = keys + (static_cast<int64_t>(row) * tokens + token) * sig_bytes;

    const int full_bytes = sig_bits_used >> 3;
    float acc = 0.0f;
    uint8_t norm_code = 0;
    if (sig_bytes == 16) {
        // Records are 16B and 16B-aligned (base ptr torch-allocated, offset a multiple of 16):
        // one uint4 load instead of 16 scalar byte loads (16x fewer L1/LSU wavefronts per warp).
        const uint4 rec = *reinterpret_cast<const uint4*>(sig);
        const unsigned int words[4] = {rec.x, rec.y, rec.z, rec.w};
        for (int byte_idx = 0; byte_idx < full_bytes; ++byte_idx) {
            const unsigned int value = (words[byte_idx >> 2] >> ((byte_idx & 3) << 3)) & 0xFFu;
            const float* xb = shared_x + (byte_idx << 3);
            acc += static_cast<float>(value & 1) * xb[0];
            acc += static_cast<float>((value >> 1) & 1) * xb[1];
            acc += static_cast<float>((value >> 2) & 1) * xb[2];
            acc += static_cast<float>((value >> 3) & 1) * xb[3];
            acc += static_cast<float>((value >> 4) & 1) * xb[4];
            acc += static_cast<float>((value >> 5) & 1) * xb[5];
            acc += static_cast<float>((value >> 6) & 1) * xb[6];
            acc += static_cast<float>((value >> 7) & 1) * xb[7];
        }
        norm_code = static_cast<uint8_t>((words[3] >> 24) & 0xFFu);
    } else {
        for (int byte_idx = 0; byte_idx < full_bytes; ++byte_idx) {
            const uint8_t value = sig[byte_idx];
            const float* xb = shared_x + (byte_idx << 3);
            acc += static_cast<float>(value & 1) * xb[0];
            acc += static_cast<float>((value >> 1) & 1) * xb[1];
            acc += static_cast<float>((value >> 2) & 1) * xb[2];
            acc += static_cast<float>((value >> 3) & 1) * xb[3];
            acc += static_cast<float>((value >> 4) & 1) * xb[4];
            acc += static_cast<float>((value >> 5) & 1) * xb[5];
            acc += static_cast<float>((value >> 6) & 1) * xb[6];
            acc += static_cast<float>((value >> 7) & 1) * xb[7];
        }
        norm_code = sig[sig_bytes - 1];
    }

    float score = 2.0f * acc - query_proj_total[row];
    if (norm_lo != nullptr) {
        score *= __expf(norm_lo[row] + static_cast<float>(norm_code) * norm_step[row]);
    }
    scores[static_cast<int64_t>(row) * tokens + token] = score;
}

// -------------------------------------------------- sampled-tail fused attention + LSE merge
// One block per (row, head): softmax attention over the m sampled tail tokens (importance
// correction pre-folded into `corr`, clip applied host-side at sampling time), LSE-merged
// IN PLACE into the main flash output. Replaces ~20 tiny torch kernels per layer with one
// launch. Dynamic smem: q (dim fp32) + logits (m fp32).
__global__ void sampled_tail_attention_merge_kernel(
    const nv_bfloat16* __restrict__ queries,      // [rows, group, dim]
    const nv_bfloat16* __restrict__ tail_keys,    // [rows, m, dim]
    const nv_bfloat16* __restrict__ tail_values,  // [rows, m, dim]
    const float* __restrict__ corr,               // [rows, m]
    nv_bfloat16* __restrict__ out_main,           // [rows, group, dim] in/out
    const float* __restrict__ lse_main,           // [rows, group]
    const float scale,
    const float clip,                             // truncated-IS cap: logit <= mean + clip; <=0 off
    const int group,
    const int m,
    const int dim) {
    const int row = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    extern __shared__ float st_smem[];
    float* q_s = st_smem;                          // [dim]
    float* logits = st_smem + dim;                 // [m]
    __shared__ float red[34];                      // warp partials + [32]=max/mean, [33]=sum

    const nv_bfloat16* q_ptr = queries + (static_cast<size_t>(row) * group + g) * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        q_s[d] = scalar_to_float(q_ptr[d]);
    }
    __syncthreads();

    // Pass 1: one warp per token, coalesced key reads, warp-reduced dot.
    const nv_bfloat16* k_base = tail_keys + static_cast<size_t>(row) * m * dim;
    const float* corr_row = corr + static_cast<size_t>(row) * m;
    for (int j = warp; j < m; j += n_warps) {
        const nv_bfloat16* k_ptr = k_base + static_cast<size_t>(j) * dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) {
            acc += q_s[d] * scalar_to_float(k_ptr[d]);
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, off);
        }
        if (lane == 0) {
            logits[j] = acc * scale + corr_row[j];
        }
    }
    __syncthreads();

    // Pass 1.5 (optional): truncated-IS clip at block mean + clip nats. The heavy importance
    // tail (-log q blowups on proposal-underscored tokens) is capped RELATIVE to the slot
    // population, jointly over the true logit and the correction (corr-only clipping measured
    // 5 points worse on fwe-32k).
    if (clip > 0.f) {
        float psum = 0.f;
        for (int j = tid; j < m; j += blockDim.x) {
            psum += logits[j];
        }
        for (int off = 16; off > 0; off >>= 1) {
            psum += __shfl_down_sync(0xffffffff, psum, off);
        }
        if (lane == 0) red[warp] = psum;
        __syncthreads();
        if (tid == 0) {
            float v = 0.f;
            for (int w = 0; w < n_warps; ++w) v += red[w];
            red[32] = v / static_cast<float>(m);
        }
        __syncthreads();
        const float cap = red[32] + clip;
        for (int j = tid; j < m; j += blockDim.x) {
            logits[j] = fminf(logits[j], cap);
        }
        __syncthreads();
    }

    // Pass 2a: block max.
    float mx = -INFINITY;
    for (int j = tid; j < m; j += blockDim.x) {
        mx = fmaxf(mx, logits[j]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    }
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (tid == 0) {
        float v = -INFINITY;
        for (int w = 0; w < n_warps; ++w) v = fmaxf(v, red[w]);
        red[32] = v;
    }
    __syncthreads();
    mx = red[32];
    __syncthreads();

    // Pass 2b: exp in place + block sum.
    float part = 0.f;
    for (int j = tid; j < m; j += blockDim.x) {
        const float w = __expf(logits[j] - mx);
        logits[j] = w;
        part += w;
    }
    for (int off = 16; off > 0; off >>= 1) {
        part += __shfl_down_sync(0xffffffff, part, off);
    }
    if (lane == 0) red[warp] = part;
    __syncthreads();
    if (tid == 0) {
        float v = 0.f;
        for (int w = 0; w < n_warps; ++w) v += red[w];
        red[33] = fmaxf(v, 1e-30f);
    }
    __syncthreads();
    const float z = red[33];
    const float lse_t = mx + __logf(z);

    // Pass 3: weighted V + in-place LSE merge; thread d owns output component d.
    const float lm = lse_main[static_cast<size_t>(row) * group + g];
    const float mmax = fmaxf(lm, lse_t);
    const float a = __expf(lm - mmax);
    const float b = __expf(lse_t - mmax);
    const float inv_ab = 1.f / (a + b);
    const nv_bfloat16* v_base = tail_values + static_cast<size_t>(row) * m * dim;
    nv_bfloat16* o_ptr = out_main + (static_cast<size_t>(row) * group + g) * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        int j = 0;
        // 4-way unroll: independent accumulators keep 4 global loads in flight per thread
        // (the serial-j loop is latency-bound at the kernel's low block count).
        for (; j + 4 <= m; j += 4) {
            acc0 += logits[j]     * scalar_to_float(v_base[static_cast<size_t>(j) * dim + d]);
            acc1 += logits[j + 1] * scalar_to_float(v_base[static_cast<size_t>(j + 1) * dim + d]);
            acc2 += logits[j + 2] * scalar_to_float(v_base[static_cast<size_t>(j + 2) * dim + d]);
            acc3 += logits[j + 3] * scalar_to_float(v_base[static_cast<size_t>(j + 3) * dim + d]);
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; j < m; ++j) {
            acc += logits[j] * scalar_to_float(v_base[static_cast<size_t>(j) * dim + d]);
        }
        const float merged = (a * scalar_to_float(o_ptr[d]) + b * (acc / z)) * inv_ab;
        o_ptr[d] = __float2bfloat16(merged);
    }
}

// ------------------------------------------ batched multi-layer cache-bypass tail KV gather
// One launch fetches the window's every layer's sampled K/V (draws are shared within a
// resample window): 32 isolated per-layer launches measured 1.4 GB/s (kernel-boundary drain
// kills PCIe pipelining) vs 8 GB/s for one batched launch. kv_ptrs[z] is the device-visible
// address of layer z's pinned [rows, tokens, 2, dim] store.
__global__ void uva_gather_kv_rows_window_kernel(
    const int32_t* __restrict__ indices,          // [rows, m]
    const int64_t* __restrict__ kv_ptrs,          // [n_layers] data ptrs (0 = skip)
    nv_bfloat16* __restrict__ out_k,              // [n_layers, rows, m, dim]
    nv_bfloat16* __restrict__ out_v,
    const int rows,
    const int m,
    const int64_t total_tokens,
    const int dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int j = blockIdx.x * n_warps + warp;
    const int row = blockIdx.y;
    const int layer = blockIdx.z;
    if (j >= m) return;
    const int64_t base = kv_ptrs[layer];
    if (base == 0) return;
    const int64_t tok = indices[static_cast<size_t>(row) * m + j];
    if (tok < 0 || tok >= total_tokens) return;
    const nv_bfloat16* cpu_kv = reinterpret_cast<const nv_bfloat16*>(base);
    const int vecs = (2 * dim * static_cast<int>(sizeof(nv_bfloat16))) / static_cast<int>(sizeof(uint4));
    const int k_vecs = vecs >> 1;
    const uint4* src = reinterpret_cast<const uint4*>(
        cpu_kv + (static_cast<size_t>(row) * total_tokens + tok) * 2 * dim);
    const size_t out_row = (static_cast<size_t>(layer) * rows + row) * m + j;
    uint4* dk = reinterpret_cast<uint4*>(out_k + out_row * dim);
    uint4* dv = reinterpret_cast<uint4*>(out_v + out_row * dim);
    for (int i = lane; i < vecs; i += 32) {
        const uint4 val = src[i];
        if (i < k_vecs) {
            dk[i] = val;
        } else {
            dv[i - k_vecs] = val;
        }
    }
}

// ------------------------------------------------------------- cache-bypass tail KV gather
// Plain UVA read of the m sampled tokens' K/V rows from the pinned host store, bypassing the
// token cache entirely: sampled tokens are one-shot draws, so caching them only pollutes the
// head's working set (and the lock/insert traffic costs more than the read).
__global__ void uva_gather_kv_rows_kernel(
    const int32_t* __restrict__ indices,          // [rows, m]
    const nv_bfloat16* __restrict__ cpu_kv,       // [rows, tokens, 2, dim] pinned (UVA)
    nv_bfloat16* __restrict__ out_k,              // [rows, m, dim]
    nv_bfloat16* __restrict__ out_v,              // [rows, m, dim]
    const int m,
    const int64_t total_tokens,
    const int dim) {
    // One WARP per token, uint4 (16B) vectorized UVA reads — matches the coalescing pattern of
    // the main gather kernels (scalar bf16 loads over PCIe measured ~15x slower). 8 warps per
    // block for memory-level parallelism across tokens; dim*2*2B assumed 16B-divisible
    // (dim=128 -> 32 uint4 per K/V pair).
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int j = blockIdx.x * n_warps + warp;
    const int row = blockIdx.y;
    if (j >= m) return;
    const int64_t tok = indices[static_cast<size_t>(row) * m + j];
    if (tok < 0 || tok >= total_tokens) return;
    const int vecs = (2 * dim * static_cast<int>(sizeof(nv_bfloat16))) / static_cast<int>(sizeof(uint4));
    const int k_vecs = vecs >> 1;
    const uint4* src = reinterpret_cast<const uint4*>(
        cpu_kv + (static_cast<size_t>(row) * total_tokens + tok) * 2 * dim);
    uint4* dk = reinterpret_cast<uint4*>(out_k + (static_cast<size_t>(row) * m + j) * dim);
    uint4* dv = reinterpret_cast<uint4*>(out_v + (static_cast<size_t>(row) * m + j) * dim);
    for (int i = lane; i < vecs; i += 32) {
        const uint4 val = src[i];
        if (i < k_vecs) {
            dk[i] = val;
        } else {
            dv[i - k_vecs] = val;
        }
    }
}

}  // namespace cometkv
