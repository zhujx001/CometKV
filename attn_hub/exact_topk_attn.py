def exact_topk_decode_attn(query_states, key_states, value_states, layer_idx, kv_cache):
    # key/value states are already committed by decode_update_kv_cache; the cache scores,
    # selects and attends internally (mirrors the cometkv_decode_attn thunk).
    return kv_cache.attn_func(query_states, layer_idx)
