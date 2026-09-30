#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace cometkv {

constexpr int kSelectThreads = 256;
constexpr int kSelectChunk = 1024;
constexpr int kSelectBinsStride = 288;  // 256 bins + min/max; keep chunk rows 128B-aligned.
constexpr int kSelectStateSize = 7;

__device__ __forceinline__ uint32_t ordered_score(float value) {
    // Match largest=True for finite values/infinities, putting NaNs first.
    if (isnan(value)) return 0xffffffffU;
    if (value == 0.f) return 0x80000000U;  // Equal-score tie handling includes signed zero.
    const uint32_t bits = __float_as_uint(value);
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}

__device__ __forceinline__ void histogram_add(int* bins, int digit, bool valid) {
    const unsigned mask = __ballot_sync(0xffffffffU, valid);
    if (valid) {
        #if __CUDA_ARCH__ >= 700
        const unsigned peers = __match_any_sync(mask, digit);
        if ((threadIdx.x & 31) == __ffs(peers) - 1)
            atomicAdd(bins + digit, __popc(peers));
        #else
        atomicAdd(bins + digit, 1);
        #endif
    }
}

template <int Pass>
__global__ void topk_histogram_kernel(
    const float* scores, int* hist, const int* state,
    int tokens, int start, int end, int hist_stride) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int shift = Pass ? state[row * kSelectStateSize + 5] : 24;
    if constexpr (Pass != 0) {
        if (state[row * kSelectStateSize + 4]) return;
    }
    const uint32_t prefix = Pass ? static_cast<uint32_t>(state[row * kSelectStateSize]) : 0;
    __shared__ int bins[256];
    __shared__ unsigned minimum, maximum;
    bins[tid] = 0;
    if constexpr (Pass == 0) {
        if (tid == 0) { minimum = 0xffffffffU; maximum = 0; }
    }
    __syncthreads();
    unsigned local_min = 0xffffffffU, local_max = 0;
    for (int i = tid; i < kSelectChunk; i += kSelectThreads) {
        const int token = start + blockIdx.y * kSelectChunk + i;
        uint32_t value = token < end ? ordered_score(scores[static_cast<size_t>(row) * tokens + token]) : 0;
        if constexpr (Pass != 0) value -= static_cast<uint32_t>(state[row * kSelectStateSize + 6]);
        bool valid = token < end;
        if constexpr (Pass == 0) {
            if (valid) { local_min = min(local_min, value); local_max = max(local_max, value); }
        }
        if constexpr (Pass != 0)
            valid = valid && (value >> (shift + 8)) == (prefix >> (shift + 8));
        histogram_add(bins, (value >> shift) & 255, valid);
    }
    if constexpr (Pass == 0) {
        for (int off = 16; off; off >>= 1) {
            local_min = min(local_min, __shfl_down_sync(0xffffffffU, local_min, off));
            local_max = max(local_max, __shfl_down_sync(0xffffffffU, local_max, off));
        }
        if ((tid & 31) == 0) { atomicMin(&minimum, local_min); atomicMax(&maximum, local_max); }
    }
    __syncthreads();
    int* out = hist + (static_cast<size_t>(row) * hist_stride + blockIdx.y) * kSelectBinsStride;
    out[tid] = bins[tid];
    if constexpr (Pass == 0) {
        if (tid == 0) { out[256] = static_cast<int>(minimum); out[257] = static_cast<int>(maximum); }
    }
}

template <int Pass>
__global__ void topk_choose_prefix_kernel(
    const int* hist, int* state, int chunks, int hist_stride, int k) {
    using Scan = cub::BlockScan<int, kSelectThreads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned bounds[16];
    __shared__ bool flat, narrow;
    const int row = blockIdx.x;
    int* params = state + row * kSelectStateSize;
    if constexpr (Pass == 0) {
        unsigned lo = 0xffffffffU, hi = 0;
        for (int c = threadIdx.x; c < chunks; c += kSelectThreads) {
            const int* bound = hist + (static_cast<size_t>(row) * hist_stride + c) * kSelectBinsStride + 256;
            lo = min(lo, static_cast<unsigned>(bound[0]));
            hi = max(hi, static_cast<unsigned>(bound[1]));
        }
        for (int off = 16; off; off >>= 1) {
            lo = min(lo, __shfl_down_sync(0xffffffffU, lo, off));
            hi = max(hi, __shfl_down_sync(0xffffffffU, hi, off));
        }
        if ((threadIdx.x & 31) == 0) {
            bounds[threadIdx.x / 32] = lo;
            bounds[8 + threadIdx.x / 32] = hi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < 8; ++w) { lo = min(lo, bounds[w]); hi = max(hi, bounds[8 + w]); }
            flat = lo == hi;
            params[4] = flat;
            // Normalize a narrow integer-key interval before taking its radix
            // digit. Subtraction handles exponent-boundary carries as well as
            // common leading bits, keeping nearly uniform scores efficient.
            const unsigned span = hi - lo;
            narrow = !flat && span < 0x1000000U;
            params[5] = narrow ? max(0, 24 - __clz(span)) : 16;
            params[6] = narrow ? static_cast<int>(lo) : 0;
            if (narrow) { params[0] = 0; params[1] = k; }
        }
        __syncthreads();
        if (flat || narrow) return;
    } else {
        if (params[4]) return;
    }
    const int digit = 255 - threadIdx.x;
    const int shift = Pass ? params[5] : 24;
    const uint32_t prefix = Pass ? static_cast<uint32_t>(params[0]) : 0;
    const int remaining = Pass ? params[1] : k;
    int count = 0;
    for (int c = 0; c < chunks; ++c)
        count += hist[(static_cast<size_t>(row) * hist_stride + c) * kSelectBinsStride + digit];
    int above;
    Scan(scan).ExclusiveSum(count, above);
    if (above < remaining && remaining <= above + count) {
        params[0] = static_cast<int>(prefix | (static_cast<uint32_t>(digit) << shift));
        params[1] = remaining - above;
    }
    if constexpr (Pass == 1) {
        if (threadIdx.x == 0) {
            params[2] = 0;
            params[3] = 0;
        }
    }
}

__device__ __forceinline__ void append_selected(
    int token, bool selected, int* counter, int* output) {
    const unsigned mask = __ballot_sync(0xffffffffU, selected);
    int base = 0;
    if ((threadIdx.x & 31) == 0 && mask) base = atomicAdd(counter, __popc(mask));
    base = __shfl_sync(0xffffffffU, base, 0);
    const unsigned preceding = (1U << (threadIdx.x & 31)) - 1U;
    if (selected) output[base + __popc(mask & preceding)] = token;
}

__global__ void topk_compact_kernel(
    const float* scores, int* output, int* candidates, int* state,
    int tokens, int output_stride, int start, int end) {
    const int row = blockIdx.x;
    int* params = state + row * kSelectStateSize;
    if (params[4]) return;
    const uint32_t prefix = static_cast<uint32_t>(params[0]);
    const uint32_t mask = 0xffffffffU << params[5];
    const uint32_t origin = static_cast<uint32_t>(params[6]);
    for (int i = threadIdx.x; i < kSelectChunk; i += kSelectThreads) {
        const int token = start + blockIdx.y * kSelectChunk + i;
        const uint32_t high = token < end
            ? (ordered_score(scores[static_cast<size_t>(row) * tokens + token]) - origin) & mask : 0;
        append_selected(token, token < end && high > prefix, params + 2,
                        output + static_cast<size_t>(row) * output_stride);
        append_selected(token, token < end && high == prefix, params + 3,
                        candidates + static_cast<size_t>(row) * tokens);
    }
}

__device__ __forceinline__ uint64_t refinement_key(const float* scores, int token, uint32_t origin) {
    // The high 16 score bits already match. Break equal-score ties by token ID,
    // so the selected set and its final token order are reproducible.
    return (static_cast<uint64_t>((ordered_score(scores[token]) - origin) & 0xffffU) << 32)
        | (0xffffffffU - static_cast<uint32_t>(token));
}

template <int Items>
__global__ void topk_refine_kernel(
    const float* scores, int* output, const int* candidates, const int* state,
    int tokens, int output_stride, int k, int start) {
    using Scan = cub::BlockScan<int, kSelectThreads>;
    using Sort = cub::BlockRadixSort<int, kSelectThreads, Items>;
    __shared__ union {
        typename Scan::TempStorage scan;
        typename Sort::TempStorage sort;
    } temp;
    __shared__ int bins[256];
    __shared__ uint64_t prefix;
    __shared__ int remaining, done, out_count;
    const int row = blockIdx.x, tid = threadIdx.x;
    const float* row_scores = scores + static_cast<size_t>(row) * tokens;
    const int* row_candidates = candidates + static_cast<size_t>(row) * tokens;
    int* row_output = output + static_cast<size_t>(row) * output_stride;
    const int* params = state + row * kSelectStateSize;
    if (params[4]) {
        for (int i = tid; i < k; i += kSelectThreads) row_output[i] = start + i;
        return;
    }
    const int count = params[3];
    const uint32_t origin = static_cast<uint32_t>(params[6]);
    if (tid == 0) {
        prefix = 0;
        remaining = params[1];
        done = remaining == count;
        out_count = params[2];
    }
    __syncthreads();
    // Usually one byte resolves the tiny boundary bucket. The full 48-bit
    // refinement also handles all-equal scores, NaNs and arbitrarily close scores.
    for (int shift = 40; shift >= 0 && !done; shift -= 8) {
        bins[tid] = 0;
        const uint64_t wanted = prefix;
        __syncthreads();
        for (int base = 0; base < count; base += kSelectThreads) {
            const int i = base + tid;
            const uint64_t key = i < count ? refinement_key(row_scores, row_candidates[i], origin) : 0;
            const bool valid = i < count && (key >> (shift + 8)) == (wanted >> (shift + 8));
            histogram_add(bins, (key >> shift) & 255, valid);
        }
        __syncthreads();
        const int digit = 255 - tid;
        const int frequency = bins[digit];
        const int rank = remaining;
        int above;
        Scan(temp.scan).ExclusiveSum(frequency, above);
        if (above < rank && rank <= above + frequency) {
            prefix = wanted | (static_cast<uint64_t>(digit) << shift);
            remaining = rank - above;
            done = frequency == rank - above;
        }
        __syncthreads();
    }
    const uint64_t threshold = prefix;
    for (int base = 0; base < count; base += kSelectThreads) {
        const int i = base + tid;
        const int token = i < count ? row_candidates[i] : 0;
        const bool selected = i < count && refinement_key(row_scores, token, origin) >= threshold;
        append_selected(token, selected, &out_count, row_output);
    }
    __syncthreads();
    int selected[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int offset = tid * Items + i;
        selected[i] = offset < k ? row_output[offset] : INT_MAX;
    }
    Sort(temp.sort).Sort(selected);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int offset = tid * Items + i;
        if (offset < k) row_output[offset] = selected[i];
    }
}

}  // namespace cometkv
