#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cometkv {

__device__ __forceinline__ int primary_cache_slot(int token, int cache_size) {
    if (cache_size <= 1) {
        return 0;
    }
    const int set_count = (cache_size + 1) >> 1;
    return (token % set_count) << 1;
}


__device__ __forceinline__ int secondary_cache_slot(int token, int cache_size) {
    return min(primary_cache_slot(token, cache_size) + 1, cache_size - 1);
}


// W-way LRU set mapping, used only when cache_stamps is provided. num_sets = cache_size / ways
// (floor): if cache_size is not divisible by ways, the trailing (cache_size % ways) slots are
// never used, so every set has exactly `ways` slots and one lock (the set's first slot).
__device__ __forceinline__ int lru_set_base(int token, int cache_size, int ways) {
    const int num_sets = max(cache_size / ways, 1);
    return (token % num_sets) * ways;
}


__device__ __forceinline__ int lru_probe_way(const int32_t* set_token_ids, int token, int ways) {
    for (int way = 0; way < ways; ++way) {
        if (set_token_ids[way] == token) {
            return way;
        }
    }
    return -1;
}


// Victim choice, called under the set lock: reuse the slot if a concurrent block already inserted
// `token`, else the first empty way, else the way with the minimum stamp (least recently used).
__device__ __forceinline__ int lru_pick_victim_way(
    const int32_t* set_token_ids,
    const int32_t* set_stamps,
    int token,
    int ways) {
    for (int way = 0; way < ways; ++way) {
        if (set_token_ids[way] == token) {
            return way;
        }
    }
    for (int way = 0; way < ways; ++way) {
        if (set_token_ids[way] < 0) {
            return way;
        }
    }
    int victim = 0;
    int min_stamp = set_stamps[0];
    for (int way = 1; way < ways; ++way) {
        const int stamp = set_stamps[way];
        if (stamp < min_stamp) {
            min_stamp = stamp;
            victim = way;
        }
    }
    return victim;
}


// Score-aware victim choice (token-cache policy "score"): the selector re-scores EVERY candidate
// every decode step, so a cached token's CURRENT priority — bucket code (smaller =
// better) or fp32 selector score (larger = better) — predicts next-step reuse better than recency
// (trace-driven sim: −23% misses at 2×topk capacity vs LRU stamps). Same-token/empty ways first
// (mirrors lru_pick_victim_way), then the worst current priority among ways NOT stamped this step
// (those are demand-hot; skipping them also closes the intra-step probe/insert window on this
// path), ties broken toward the older stamp. If every way was stamped this step, fall back to LRU.
__device__ __forceinline__ int scored_pick_victim_way(
    const int32_t* set_token_ids_in,
    const int32_t* set_stamps_in,
    int token,
    int ways,
    const int16_t* prio_buckets_row,
    const float* prio_scores_row,
    int current_step) {
    // volatile: called under the set lock, but the caller's unlocked hit-probe already pulled
    // these lines into L1; a concurrent block's insert only reaches L2 (threadfence before its
    // unlock), so the scan must bypass L1 to see it.
    volatile const int32_t* set_token_ids = set_token_ids_in;
    volatile const int32_t* set_stamps = set_stamps_in;
    for (int way = 0; way < ways; ++way) {
        if (set_token_ids[way] == token) {
            return way;
        }
    }
    for (int way = 0; way < ways; ++way) {
        if (set_token_ids[way] < 0) {
            return way;
        }
    }
    int victim = -1;
    float worst_badness = 0.0f;
    int worst_stamp = 0;
    for (int way = 0; way < ways; ++way) {
        const int stamp = set_stamps[way];
        if (stamp == current_step) {
            continue;
        }
        const int id = set_token_ids[way];
        const float badness = prio_buckets_row != nullptr
                                  ? static_cast<float>(prio_buckets_row[id])
                                  : -prio_scores_row[id];
        if (victim < 0 || badness > worst_badness
            || (badness == worst_badness && stamp < worst_stamp)) {
            victim = way;
            worst_badness = badness;
            worst_stamp = stamp;
        }
    }
    if (victim >= 0) {
        return victim;
    }
    return lru_pick_victim_way(set_token_ids_in, set_stamps_in, token, ways);
}


__device__ __forceinline__ int fixed_window_start(int decode_count, int recent_capacity, int slide_stride) {
    if (decode_count <= slide_stride) {
        return 0;
    }
    const int recent_overlap = max(recent_capacity - slide_stride, 0);
    return ((decode_count - slide_stride - 1) / slide_stride + 1) * slide_stride - recent_overlap;
}


template <bool AdvanceVisibleLength>
__global__ void append_lockstep_local_kv_cache_kernel(
    const uint16_t* key_states,
    const uint16_t* value_states,
    uint16_t* local_keys,
    uint16_t* local_values,
    int32_t* visible_lengths,
    int batch_size,
    int kv_heads,
    int local_tokens,
    int local_capacity,
    int slide_stride,
    int decode_step,
    int dim,
    const int32_t* decode_step_ptr = nullptr) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= batch_size * kv_heads) {
        return;
    }
    // CUDA-graph mode: read the decode step from a device counter so the captured kernel writes the
    // correct (advancing) slot on every replay instead of the slot baked at capture time.
    if (decode_step_ptr != nullptr) {
        decode_step = decode_step_ptr[0];
    }

    const int batch_idx = row / kv_heads;
    const int head_idx = row - batch_idx * kv_heads;
    const int old_start = fixed_window_start(decode_step, local_capacity, slide_stride);
    const int new_start = fixed_window_start(decode_step + 1, local_capacity, slide_stride);
    const int shift = max(new_start - old_start, 0);
    const int prev_len = min(max(decode_step - old_start, 0), local_capacity);
    const int keep = max(prev_len - shift, 0);
    const int slot = decode_step - new_start;
    if (slot < 0 || slot >= local_capacity || local_capacity > local_tokens) {
        return;
    }

    const int row_base_token = row * local_tokens;
    if (shift > 0 && keep > 0) {
        if (dim == 128) {
            constexpr int vec_count = 16;
            uint4* local_key_vec = reinterpret_cast<uint4*>(local_keys);
            uint4* local_value_vec = reinterpret_cast<uint4*>(local_values);
            const int row_vec_base = row_base_token * vec_count;
            for (int token = 0; token < keep; ++token) {
                const int src_vec_offset = row_vec_base + (token + shift) * vec_count;
                const int dst_vec_offset = row_vec_base + token * vec_count;
                for (int vec = lane; vec < vec_count; vec += blockDim.x) {
                    local_key_vec[dst_vec_offset + vec] = local_key_vec[src_vec_offset + vec];
                    local_value_vec[dst_vec_offset + vec] = local_value_vec[src_vec_offset + vec];
                }
            }
        } else {
            const int row_scalar_base = row_base_token * dim;
            for (int token = 0; token < keep; ++token) {
                const int src_offset = row_scalar_base + (token + shift) * dim;
                const int dst_offset = row_scalar_base + token * dim;
                for (int col = lane; col < dim; col += blockDim.x) {
                    local_keys[dst_offset + col] = local_keys[src_offset + col];
                    local_values[dst_offset + col] = local_values[src_offset + col];
                }
            }
        }
    }

    if (dim == 128) {
        constexpr int vec_count = 16;
        const uint4* key_vec = reinterpret_cast<const uint4*>(key_states);
        const uint4* value_vec = reinterpret_cast<const uint4*>(value_states);
        uint4* local_key_vec = reinterpret_cast<uint4*>(local_keys);
        uint4* local_value_vec = reinterpret_cast<uint4*>(local_values);
        const int src_vec_offset = (batch_idx * kv_heads + head_idx) * vec_count;
        const int dst_vec_offset = (row_base_token + slot) * vec_count;
        if (lane < vec_count) {
            local_key_vec[dst_vec_offset + lane] = key_vec[src_vec_offset + lane];
            local_value_vec[dst_vec_offset + lane] = value_vec[src_vec_offset + lane];
        }
        if constexpr (AdvanceVisibleLength) {
            if (head_idx == 0 && lane == 0) {
                visible_lengths[batch_idx] += 1;
            }
        }
        return;
    }

    const int src_offset = (batch_idx * kv_heads + head_idx) * dim;
    const int dst_offset = (row_base_token + slot) * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        local_keys[dst_offset + col] = key_states[src_offset + col];
        local_values[dst_offset + col] = value_states[src_offset + col];
    }
    if constexpr (AdvanceVisibleLength) {
        if (head_idx == 0 && lane == 0) {
            visible_lengths[batch_idx] += 1;
        }
    }
}

__global__ void refresh_static_prompt_recent_state_kernel(
    const int32_t* prompt_lengths,
    const int32_t* visible_lengths,
    const uint16_t* prompt_recent_keys,
    const uint16_t* prompt_recent_values,
    uint16_t* static_keys,
    uint16_t* static_values,
    int32_t* static_lengths,
    int batch_size,
    int kv_heads,
    int static_pattern_start,
    int static_pattern_end,
    int dim) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int rows = batch_size * kv_heads;
    if (row >= rows) {
        return;
    }

    const int batch_idx = row / kv_heads;
    const int head_idx = row - batch_idx * kv_heads;
    const int prompt_length = prompt_lengths[batch_idx];
    const int visible_length = visible_lengths[batch_idx];
    const int sink_end = min(prompt_length, static_pattern_start);
    const int recent_start = max(sink_end, visible_length - static_pattern_end);
    int prompt_recent_count = max(min(prompt_length, visible_length) - recent_start, 0);
    prompt_recent_count = min(prompt_recent_count, static_pattern_end);
    const int static_length = sink_end + prompt_recent_count;

    if (lane == 0) {
        static_lengths[row] = static_length;
    }
    if (prompt_recent_count <= 0) {
        return;
    }

    const int src_start = static_pattern_end - prompt_recent_count;
    const int dst_start = static_pattern_start;
    const int static_total = static_pattern_start + static_pattern_end;
    const int row_3d = batch_idx * kv_heads + head_idx;

    if (dim == 128) {
        constexpr int vec_count = 16;
        const uint4* prompt_key_vec = reinterpret_cast<const uint4*>(prompt_recent_keys);
        const uint4* prompt_value_vec = reinterpret_cast<const uint4*>(prompt_recent_values);
        uint4* static_key_vec = reinterpret_cast<uint4*>(static_keys);
        uint4* static_value_vec = reinterpret_cast<uint4*>(static_values);
        for (int token = 0; token < prompt_recent_count; ++token) {
            const int src_token = src_start + token;
            const int dst_token = dst_start + token;
            const int src_vec_offset = (row_3d * static_pattern_end + src_token) * vec_count;
            const int dst_vec_offset = (row_3d * static_total + dst_token) * vec_count;
            if (lane < vec_count) {
                static_key_vec[dst_vec_offset + lane] = prompt_key_vec[src_vec_offset + lane];
                static_value_vec[dst_vec_offset + lane] = prompt_value_vec[src_vec_offset + lane];
            }
        }
        return;
    }

    for (int token = 0; token < prompt_recent_count; ++token) {
        const int src_token = src_start + token;
        const int dst_token = dst_start + token;
        const int src_offset = (row_3d * static_pattern_end + src_token) * dim;
        const int dst_offset = (row_3d * static_total + dst_token) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            static_keys[dst_offset + col] = prompt_recent_keys[src_offset + col];
            static_values[dst_offset + col] = prompt_recent_values[src_offset + col];
        }
    }
}


__global__ void concat_static_recent_lookup_gather_uva_kv_update_cache_kernel(
    const uint16_t* static_keys,
    const uint16_t* static_values,
    const uint16_t* recent_keys,
    const uint16_t* recent_values,
    const int32_t* request_token_ids,
    const uint16_t* cpu_kv,
    uint16_t* out_keys,
    uint16_t* out_values,
    int32_t* hit_mask,
    int32_t* cache_token_ids,
    int32_t* cache_locks,
    uint16_t* cache_keys,
    uint16_t* cache_values,
    int rows,
    int static_capacity,
    int recent_capacity,
    int request_count,
    int total_tokens,
    int cache_size,
    int static_len,
    int recent_len,
    int sparse_len,
    int total_len,
    int out_row_len,
    int dim,
    int32_t* cache_stamps = nullptr,
    const int32_t* step_ptr = nullptr,
    int ways = 2,
    const int16_t* prio_buckets = nullptr,
    const float* prio_scores = nullptr,
    int prio_stride = 0,
    int16_t* cache_prio = nullptr) {
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
        const uint4* cpu_vec = reinterpret_cast<const uint4*>(cpu_kv);
        uint4* out_key_vec = reinterpret_cast<uint4*>(out_keys);
        uint4* out_value_vec = reinterpret_cast<uint4*>(out_values);
        uint4* cache_key_vec = reinterpret_cast<uint4*>(cache_keys);
        uint4* cache_value_vec = reinterpret_cast<uint4*>(cache_values);
        // out rows may be wider than the produced [static|recent|sparse] prefix (out_row_len >=
        // total_len) so callers can gather straight into a fixed-width concat buffer.
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

        const int hit_offset = row * request_count + request_idx;
        const int token = request_token_ids[hit_offset];
        if (token < 0) {
            const uint4 zero = make_uint4(0, 0, 0, 0);
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = zero;
                out_value_vec[out_vec_offset + lane] = zero;
            }
            if (lane == 0) {
                hit_mask[hit_offset] = 1;
            }
            return;
        }

        const int row_base = row * cache_size;
        // Two cache-replacement regimes (bit-identical old behavior when cache_stamps==nullptr):
        //   stamps==nullptr: legacy 2-way set-associative always-insert (primary/secondary).
        //   stamps!=nullptr: W-way set-associative with LRU victim choice (lru_* helpers).
        const bool lru = cache_stamps != nullptr;
        const int set_base = lru ? lru_set_base(token, cache_size, ways) : 0;
        const int primary_slot = lru ? 0 : primary_cache_slot(token, cache_size);
        const int secondary_slot = lru ? 0 : secondary_cache_slot(token, cache_size);
        int slot;
        bool hit;
        if (lru) {
            const int way = lru_probe_way(cache_token_ids + row_base + set_base, token, ways);
            hit = way >= 0;
            slot = set_base + (hit ? way : 0);
        } else {
            slot = primary_slot;
            hit = cache_token_ids[row_base + primary_slot] == token;
            if (!hit && cache_token_ids[row_base + secondary_slot] == token) {
                hit = true;
                slot = secondary_slot;
            }
        }

        if (lane == 0) {
            hit_mask[hit_offset] = hit ? 1 : 0;
        }

        if (hit) {
            const int cache_vec_offset = (row_base + slot) * vec_count;
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = cache_key_vec[cache_vec_offset + lane];
                out_value_vec[out_vec_offset + lane] = cache_value_vec[cache_vec_offset + lane];
            }
            if (lru && lane == 0 && step_ptr != nullptr) {
                cache_stamps[row_base + slot] = step_ptr[0];  // touch: mark recently used
                if (cache_prio != nullptr && prio_buckets != nullptr) {
                    cache_prio[row_base + slot] =
                        prio_buckets[static_cast<int64_t>(row) * prio_stride + token];
                }
            }
            return;
        }

        const int src_vec_offset = ((row * total_tokens + token) * 2) * vec_count;
        uint4 key_vec;
        uint4 value_vec;
        if (lane < vec_count) {
            key_vec = cpu_vec[src_vec_offset + lane];
            value_vec = cpu_vec[src_vec_offset + vec_count + lane];
            out_key_vec[out_vec_offset + lane] = key_vec;
            out_value_vec[out_vec_offset + lane] = value_vec;
        }
        __syncthreads();

        __shared__ int selected_slot_vec;
        if (lane == 0) {
            if (lru) {
                // One lock per set (the set's first slot); pick the LRU/empty victim under it.
                // With a priority buffer ("score" policy) the victim is the worst-current-score
                // way instead of the least-recently-used one.
                int32_t* set_lock = cache_locks + row_base + set_base;
                while (atomicCAS(set_lock, 0, 1) != 0) {
                }
                const bool scored = prio_buckets != nullptr || prio_scores != nullptr;
                const int victim = scored
                    ? scored_pick_victim_way(
                          cache_token_ids + row_base + set_base,
                          cache_stamps + row_base + set_base, token, ways,
                          prio_buckets == nullptr
                              ? nullptr
                              : prio_buckets + static_cast<int64_t>(row) * prio_stride,
                          prio_scores == nullptr
                              ? nullptr
                              : prio_scores + static_cast<int64_t>(row) * prio_stride,
                          step_ptr == nullptr ? 0 : step_ptr[0])
                    : lru_pick_victim_way(
                          cache_token_ids + row_base + set_base,
                          cache_stamps + row_base + set_base, token, ways);
                selected_slot_vec = set_base + victim;
            } else {
                int32_t* primary_lock = cache_locks + row_base + primary_slot;
                while (atomicCAS(primary_lock, 0, 1) != 0) {
                }
                const int primary_token = cache_token_ids[row_base + primary_slot];
                if (primary_token < 0 || primary_token == token) {
                    selected_slot_vec = primary_slot;
                } else {
                    atomicExch(primary_lock, 0);
                    int32_t* secondary_lock = cache_locks + row_base + secondary_slot;
                    while (atomicCAS(secondary_lock, 0, 1) != 0) {
                    }
                    selected_slot_vec = secondary_slot;
                }
            }
        }
        __syncthreads();

        const int cache_vec_offset = (row_base + selected_slot_vec) * vec_count;
        if (lane < vec_count) {
            cache_key_vec[cache_vec_offset + lane] = key_vec;
            cache_value_vec[cache_vec_offset + lane] = value_vec;
        }
        __syncthreads();

        if (lane == 0) {
            cache_token_ids[row_base + selected_slot_vec] = token;
            if (lru && step_ptr != nullptr) {
                cache_stamps[row_base + selected_slot_vec] = step_ptr[0];
                if (cache_prio != nullptr && prio_buckets != nullptr) {
                    cache_prio[row_base + selected_slot_vec] =
                        prio_buckets[static_cast<int64_t>(row) * prio_stride + token];
                }
            }
            __threadfence();
            // Release the lock actually held: the set lock (LRU) or the chosen slot lock (2-way).
            atomicExch(cache_locks + row_base + (lru ? set_base : selected_slot_vec), 0);
        }
        return;
    }

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

    const int hit_offset = row * request_count + request_idx;
    const int token = request_token_ids[hit_offset];
    if (token < 0) {
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = 0;
            out_values[out_offset + col] = 0;
        }
        if (lane == 0) {
            hit_mask[hit_offset] = 1;
        }
        return;
    }

    const int row_base = row * cache_size;
    const int primary_slot = primary_cache_slot(token, cache_size);
    const int secondary_slot = secondary_cache_slot(token, cache_size);
    int slot = primary_slot;
    bool hit = cache_token_ids[row_base + primary_slot] == token;
    if (!hit && cache_token_ids[row_base + secondary_slot] == token) {
        hit = true;
        slot = secondary_slot;
    }

    if (lane == 0) {
        hit_mask[hit_offset] = hit ? 1 : 0;
    }

    if (hit) {
        const int cache_offset = (row_base + slot) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = cache_keys[cache_offset + col];
            out_values[out_offset + col] = cache_values[cache_offset + col];
        }
        return;
    }

    const int src_offset = ((row * total_tokens + token) * 2) * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        const uint16_t key_value = cpu_kv[src_offset + col];
        const uint16_t value_value = cpu_kv[src_offset + dim + col];
        out_keys[out_offset + col] = key_value;
        out_values[out_offset + col] = value_value;
    }
    __syncthreads();

    __shared__ int selected_slot;
    if (lane == 0) {
        int32_t* primary_lock = cache_locks + row_base + primary_slot;
        while (atomicCAS(primary_lock, 0, 1) != 0) {
        }
        const int primary_token = cache_token_ids[row_base + primary_slot];
        if (primary_token < 0 || primary_token == token) {
            selected_slot = primary_slot;
        } else {
            atomicExch(primary_lock, 0);
            int32_t* secondary_lock = cache_locks + row_base + secondary_slot;
            while (atomicCAS(secondary_lock, 0, 1) != 0) {
            }
            selected_slot = secondary_slot;
        }
    }
    __syncthreads();

    const int cache_offset = (row_base + selected_slot) * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        cache_keys[cache_offset + col] = out_keys[out_offset + col];
        cache_values[cache_offset + col] = out_values[out_offset + col];
    }
    __syncthreads();

    if (lane == 0) {
        cache_token_ids[row_base + selected_slot] = token;
        __threadfence();
        atomicExch(cache_locks + row_base + selected_slot, 0);
    }
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to(float v);
template <>
__device__ __forceinline__ half float_to<half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ nv_bfloat16 float_to<nv_bfloat16>(float v) { return __float2bfloat16(v); }


// int8 variant of the concat gather: cpu_kv is stored as int8 to halve PCIe/UVA traffic on the
// (cache-miss) demand path. K is dequantized per-channel with k_scale[row, channel] (kept on GPU,
// frozen at prefill) and V per-token with v_scale[row, token] (pinned, read over UVA alongside the
// codes). static/recent regions and the GPU token cache stay in scalar_t (bf16/fp16) -- only the
// CPU-resident retrieval store is int8, and it is dequantized to scalar_t as it is gathered, so the
// FlashAttention input and the on-GPU token cache remain full bf16/fp16.
template <typename scalar_t>
__global__ void concat_static_recent_lookup_gather_uva_kv_update_cache_int8_kernel(
    const scalar_t* static_keys,
    const scalar_t* static_values,
    const scalar_t* recent_keys,
    const scalar_t* recent_values,
    const int32_t* request_token_ids,
    const int8_t* cpu_kv_int8,
    const float* k_scale,
    const float* v_scale,
    scalar_t* out_keys,
    scalar_t* out_values,
    int32_t* hit_mask,
    int32_t* cache_token_ids,
    int32_t* cache_locks,
    scalar_t* cache_keys,
    scalar_t* cache_values,
    int rows,
    int static_capacity,
    int recent_capacity,
    int request_count,
    int total_tokens,
    int cache_size,
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

    // Mirror the (production-stable) bf16 kernel's dim==128 vectorized structure exactly: 16 active
    // lanes, uint4 traffic, and a register-based cache write (no global re-read, no 128-thread spin).
    // The only int8-specific change is the cache-miss read, which dequantizes from int8 instead of
    // loading bf16 directly. This keeps the inter-block cache-update timing identical to bf16.
    if (dim == 128) {
        constexpr int vec_count = 16;   // 16 lanes * 8 scalar_t = 128
        const uint4* static_key_vec = reinterpret_cast<const uint4*>(static_keys);
        const uint4* static_value_vec = reinterpret_cast<const uint4*>(static_values);
        const uint4* recent_key_vec = reinterpret_cast<const uint4*>(recent_keys);
        const uint4* recent_value_vec = reinterpret_cast<const uint4*>(recent_values);
        uint4* out_key_vec = reinterpret_cast<uint4*>(out_keys);
        uint4* out_value_vec = reinterpret_cast<uint4*>(out_values);
        uint4* cache_key_vec = reinterpret_cast<uint4*>(cache_keys);
        uint4* cache_value_vec = reinterpret_cast<uint4*>(cache_values);
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
        const int hit_offset = row * request_count + request_idx;
        const int token = request_token_ids[hit_offset];
        if (token < 0) {
            const uint4 zero = make_uint4(0, 0, 0, 0);
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = zero;
                out_value_vec[out_vec_offset + lane] = zero;
            }
            if (lane == 0) {
                hit_mask[hit_offset] = 1;
            }
            return;
        }

        const int row_base = row * cache_size;
        const int primary_slot = primary_cache_slot(token, cache_size);
        const int secondary_slot = secondary_cache_slot(token, cache_size);
        int slot = primary_slot;
        bool hit = cache_token_ids[row_base + primary_slot] == token;
        if (!hit && cache_token_ids[row_base + secondary_slot] == token) {
            hit = true;
            slot = secondary_slot;
        }
        if (lane == 0) {
            hit_mask[hit_offset] = hit ? 1 : 0;
        }
        if (hit) {
            const int cache_vec_offset = (row_base + slot) * vec_count;
            if (lane < vec_count) {
                out_key_vec[out_vec_offset + lane] = cache_key_vec[cache_vec_offset + lane];
                out_value_vec[out_vec_offset + lane] = cache_value_vec[cache_vec_offset + lane];
            }
            return;
        }

        // Miss: dequant the 8 channels this lane owns into a uint4 register, write out + (later) cache.
        uint4 key_vec;
        uint4 value_vec;
        if (lane < vec_count) {
            const int c0 = lane * 8;
            const int k_code_base = ((row * total_tokens + token) * 2 + 0) * dim + c0;
            const int v_code_base = ((row * total_tokens + token) * 2 + 1) * dim + c0;
            const int k_scale_base = row * dim + c0;
            const float v_token_scale = v_scale[row * total_tokens + token];
            union { scalar_t s[8]; uint4 v; } kpack, vpack;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                kpack.s[i] = float_to<scalar_t>(static_cast<float>(cpu_kv_int8[k_code_base + i]) * k_scale[k_scale_base + i]);
                vpack.s[i] = float_to<scalar_t>(static_cast<float>(cpu_kv_int8[v_code_base + i]) * v_token_scale);
            }
            key_vec = kpack.v;
            value_vec = vpack.v;
            out_key_vec[out_vec_offset + lane] = key_vec;
            out_value_vec[out_vec_offset + lane] = value_vec;
        }
        __syncthreads();

        __shared__ int selected_slot_vec;
        if (lane == 0) {
            int32_t* primary_lock = cache_locks + row_base + primary_slot;
            while (atomicCAS(primary_lock, 0, 1) != 0) {
            }
            const int primary_token = cache_token_ids[row_base + primary_slot];
            if (primary_token < 0 || primary_token == token) {
                selected_slot_vec = primary_slot;
            } else {
                atomicExch(primary_lock, 0);
                int32_t* secondary_lock = cache_locks + row_base + secondary_slot;
                while (atomicCAS(secondary_lock, 0, 1) != 0) {
                }
                selected_slot_vec = secondary_slot;
            }
        }
        __syncthreads();

        const int cache_vec_offset = (row_base + selected_slot_vec) * vec_count;
        if (lane < vec_count) {
            cache_key_vec[cache_vec_offset + lane] = key_vec;
            cache_value_vec[cache_vec_offset + lane] = value_vec;
        }
        __syncthreads();

        if (lane == 0) {
            cache_token_ids[row_base + selected_slot_vec] = token;
            __threadfence();
            atomicExch(cache_locks + row_base + selected_slot_vec, 0);
        }
        return;
    }

    // Generic-dim scalar fallback (not used at head_dim==128).
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

    const int hit_offset = row * request_count + request_idx;
    const int token = request_token_ids[hit_offset];
    if (token < 0) {
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = float_to<scalar_t>(0.0f);
            out_values[out_offset + col] = float_to<scalar_t>(0.0f);
        }
        if (lane == 0) {
            hit_mask[hit_offset] = 1;
        }
        return;
    }

    const int row_base = row * cache_size;
    const int primary_slot = primary_cache_slot(token, cache_size);
    const int secondary_slot = secondary_cache_slot(token, cache_size);
    int slot = primary_slot;
    bool hit = cache_token_ids[row_base + primary_slot] == token;
    if (!hit && cache_token_ids[row_base + secondary_slot] == token) {
        hit = true;
        slot = secondary_slot;
    }
    if (lane == 0) {
        hit_mask[hit_offset] = hit ? 1 : 0;
    }
    if (hit) {
        const int cache_offset = (row_base + slot) * dim;
        for (int col = lane; col < dim; col += blockDim.x) {
            out_keys[out_offset + col] = cache_keys[cache_offset + col];
            out_values[out_offset + col] = cache_values[cache_offset + col];
        }
        return;
    }

    const int k_code_base = ((row * total_tokens + token) * 2 + 0) * dim;
    const int v_code_base = ((row * total_tokens + token) * 2 + 1) * dim;
    const float v_token_scale = v_scale[row * total_tokens + token];
    const int k_scale_base = row * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        const float k_val = static_cast<float>(cpu_kv_int8[k_code_base + col]) * k_scale[k_scale_base + col];
        const float v_val = static_cast<float>(cpu_kv_int8[v_code_base + col]) * v_token_scale;
        out_keys[out_offset + col] = float_to<scalar_t>(k_val);
        out_values[out_offset + col] = float_to<scalar_t>(v_val);
    }
    __syncthreads();

    __shared__ int selected_slot;
    if (lane == 0) {
        int32_t* primary_lock = cache_locks + row_base + primary_slot;
        while (atomicCAS(primary_lock, 0, 1) != 0) {
        }
        const int primary_token = cache_token_ids[row_base + primary_slot];
        if (primary_token < 0 || primary_token == token) {
            selected_slot = primary_slot;
        } else {
            atomicExch(primary_lock, 0);
            int32_t* secondary_lock = cache_locks + row_base + secondary_slot;
            while (atomicCAS(secondary_lock, 0, 1) != 0) {
            }
            selected_slot = secondary_slot;
        }
    }
    __syncthreads();

    const int cache_offset = (row_base + selected_slot) * dim;
    for (int col = lane; col < dim; col += blockDim.x) {
        cache_keys[cache_offset + col] = out_keys[out_offset + col];
        cache_values[cache_offset + col] = out_values[out_offset + col];
    }
    __syncthreads();

    if (lane == 0) {
        cache_token_ids[row_base + selected_slot] = token;
        __threadfence();
        atomicExch(cache_locks + row_base + selected_slot, 0);
    }
}


}  // namespace cometkv
