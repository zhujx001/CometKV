def cometkv_decode_attn(query_states, key_states, value_states, layer_idx, cometkv_cache):
    return cometkv_cache.attn_func(query_states.contiguous(), layer_idx)
