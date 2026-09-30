import sys
from pathlib import Path

KERNEL_LIB = Path(__file__).resolve().parents[1]
if str(KERNEL_LIB) not in sys.path:
    sys.path.insert(0, str(KERNEL_LIB))

import cometkv


def test_cometkv_gather_exports_fast_path_kernels_only():
    assert hasattr(cometkv, "append_lockstep_local_kv_cache")
    assert hasattr(cometkv, "append_lockstep_local_kv_cache_and_advance")
    assert hasattr(cometkv, "refresh_static_prompt_recent_state")
    assert hasattr(cometkv, "concat_static_recent_lookup_gather_uva_kv_update_cache")

    assert not hasattr(cometkv, "append_recent_kv_cache")
    assert not hasattr(cometkv, "append_fixed_window_kv_cache")
    assert not hasattr(cometkv, "lookup_token_cache_lru")
    assert not hasattr(cometkv, "lookup_token_cache_protected_lru")
    assert not hasattr(cometkv, "concat_static_recent_sparse_kv")
    assert not hasattr(cometkv, "gather_copy_vectors")
    assert not hasattr(cometkv, "batch_gemm_softmax")
    assert not hasattr(cometkv, "WaveBufferCPU")
    assert not hasattr(cometkv, "append_lockstep_local_kv_cache_and_advance_dev")
    assert not hasattr(cometkv, "token_cache_" + "pre" + "fetch_fill")


def test_cometkv_signature_exports_fast_path_kernels_only():
    assert hasattr(cometkv, "asym_signature_score_into")
    assert hasattr(cometkv, "grouped_signature_score_into")
    assert hasattr(cometkv, "exact_topk_indices_into")
    assert hasattr(cometkv, "sampled_tail_attention_merge")
    assert hasattr(cometkv, "uva_gather_kv_rows")
    assert hasattr(cometkv, "uva_gather_kv_rows_window")

    assert not hasattr(cometkv, "build_packed_signatures")
    assert not hasattr(cometkv, "build_packed_signatures_into")
    removed_exports = (
        "build_grouped_query_signatures_into",
        "grouped_query_" + "ham" + "ming_topk_scratch_into",
        "ham" + "ming_assign",
        "ham" + "ming_majority_vote",
        "ham" + "ming_assign_and_vote",
        "ham" + "ming_topk",
        "ham" + "ming_topk_scratch_into",
        "ham" + "ming_topk_with_values",
        "q" + "proj_signature_score_into",
        "q" + "pair_signature_score_into",
        "q" + "proj_fused_bucket_topk",
        "q" + "proj_fused_bucket_topk_" + "pre" + "fetch",
    )
    for name in removed_exports:
        assert not hasattr(cometkv, name)
