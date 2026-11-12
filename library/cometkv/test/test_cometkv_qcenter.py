import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CometKV runtime tests require a CUDA GPU.", allow_module_level=True)

from cache_hub.cometkv_cache import cometkv_cache

DTYPE = torch.bfloat16
DEVICE = "cuda:0"
SEQ = 96
# Block-0 statistics: the prompt minus the 4 sink tokens (never retrieved). The 4 recent tokens
# stay in, since they become retrieval candidates at the first window slide.
BLOCK0 = slice(4, SEQ)
RECENT = slice(SEQ - 4, SEQ)


def _make_cache(monkeypatch, qcenter=None, **env):
    for name in ("COMETKV_QCENTER", "COMETKV_SIG_BITS_USED", "COMETKV_QUERY_AGG", "COMETKV_NO_KEY_CENTER"):
        monkeypatch.delenv(name, raising=False)
    if qcenter is not None:
        monkeypatch.setenv("COMETKV_QCENTER", qcenter)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("COMETKV_SAMPLE_FRAC", "0")
    return cometkv_cache(
        valid_start=np.array([0], dtype=np.int32), layer_num=1, batch_size=1, max_length=128,
        num_key_value_heads=2, num_heads=4, head_dim=128, dtype=DTYPE, layer_mapping={"0": DEVICE},
        max_new_length=8, static_pattern_start=4, static_pattern_end=4, retrieval_budget=0.1,
        sig_bits=128, sig_topk=12, sig_chunk_size=16384, sig_seed=1234, sig_mode="random_orth",
        sig_token_cache_size=16, prefill_bsz=1, num_gpus=1, model_size=8, sig_min_retrieval_topk=0,
        query_aggregation="q_sum",
    )


def _prefill(cache, seed=3, sink_scale=1.0, recent_scale=1.0):
    torch.manual_seed(seed)
    # A shared query offset, as in real models (query cones), so mu_q is far from zero.
    q = torch.randn((1, SEQ, 4, 128), dtype=torch.float32, device=DEVICE) + 2.0
    k = torch.randn((1, SEQ, 2, 128), dtype=DTYPE, device=DEVICE)
    # Optional outlier sink / recent tokens (e.g. attention-sink keys).
    q[:, :4] *= sink_scale
    k[:, :4] *= sink_scale
    q[:, -4:] *= recent_scale
    k[:, -4:] *= recent_scale
    v = torch.randn_like(k)
    cache.prefill_update_kv_cache(q.to(DTYPE), k, v, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)
    torch.cuda.synchronize()
    return q.to(DTYPE), k


def test_qcenter_auto_defaults_to_8bit_with_112_bits(monkeypatch):
    cache = _make_cache(monkeypatch)
    assert cache.qcenter == "8bit"
    assert cache.asym_sig_bits == 112


def test_qcenter_auto_turns_off_without_its_preconditions(monkeypatch):
    cache = _make_cache(monkeypatch, COMETKV_NO_KEY_CENTER="1")
    assert cache.qcenter == "off"
    assert cache.asym_sig_bits == 120


def test_qcenter_off_keeps_the_120_bit_scorer(monkeypatch):
    cache = _make_cache(monkeypatch, qcenter="off")
    assert cache.qcenter == "off"
    assert cache.asym_sig_bits == 120


def test_qcenter_rejects_unknown_mode(monkeypatch):
    with pytest.raises(ValueError, match="COMETKV_QCENTER"):
        _make_cache(monkeypatch, qcenter="on")


def test_qcenter_fp_stores_exact_query_mean_and_bias(monkeypatch):
    cache = _make_cache(monkeypatch, qcenter="fp")
    q, k = _prefill(cache)
    # mu_q: mean over the prompt of the GQA-summed query, one per (row, kv head).
    q_sum = q.float().reshape(1, SEQ, 2, 2, 128).sum(dim=3)                 # [1, T, kv, d]
    mu_q = q_sum[:, BLOCK0].mean(dim=1).reshape(2, 128)
    torch.testing.assert_close(cache.q_mean[0], mu_q, rtol=1e-4, atol=1e-3)
    # c = mu_q . (k - mu_block) for every encoded prompt key (mu_block = offloaded prompt key mean).
    keys = k.float().permute(0, 2, 1, 3).reshape(2, SEQ, 128)
    mu_k = keys[:, BLOCK0].mean(dim=1, keepdim=True)
    expected = torch.einsum("rtd,rd->rt", keys - mu_k, mu_q)
    torch.testing.assert_close(cache.q_bias[0][:, :SEQ], expected, rtol=1e-3, atol=2e-2)


def test_qcenter_8bit_bias_is_within_one_quantization_step(monkeypatch):
    fp = _make_cache(monkeypatch, qcenter="fp")
    _prefill(fp)
    exact = fp.q_bias[0][:, BLOCK0].clone()
    q8 = _make_cache(monkeypatch, qcenter="8bit")
    _prefill(q8)
    coded = q8._qbias_from_signatures(0)[:, BLOCK0]
    step = (exact.amax(dim=1, keepdim=True) - exact.amin(dim=1, keepdim=True)) / 255.0
    assert ((coded - exact).abs() <= step * 0.5 + 1e-3).all()


def test_qcenter_8bit_reads_c_from_the_signature_without_an_fp32_buffer(monkeypatch):
    # 16 B/token: 112 sign bits, the 8-bit c code in byte 14 and the norm code in byte 15.
    cache = _make_cache(monkeypatch)
    assert cache.qcenter == "8bit" and cache.q_bias is None
    assert cache.signature_index[0].size(-1) == 16


@pytest.mark.parametrize("qcenter", ["fp", "8bit"])
def test_block0_statistics_ignore_the_sink(monkeypatch, qcenter):
    cache = _make_cache(monkeypatch, qcenter=qcenter)
    q, k = _prefill(cache, sink_scale=30.0)
    keys = k.float().permute(0, 2, 1, 3).reshape(2, SEQ, 128)
    mu_k = keys[:, BLOCK0].mean(dim=1)
    torch.testing.assert_close(cache.sig_key_mean[0], mu_k, rtol=1e-4, atol=1e-3)
    mu_q = q.float().reshape(1, SEQ, 2, 2, 128).sum(dim=3)[:, BLOCK0].mean(dim=1).reshape(2, 128)
    torch.testing.assert_close(cache.q_mean[0], mu_q, rtol=1e-4, atol=1e-3)
    # The 8-bit log-norm range spans block 0 (recent included), not the outlier sink.
    log_norms = (keys[:, BLOCK0] - mu_k[:, None]).norm(dim=-1).log()
    torch.testing.assert_close(cache.sig_norm_lo[0], log_norms.amin(dim=1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cache.sig_norm_step[0], (log_norms.amax(dim=1) - log_norms.amin(dim=1)) / 255.0,
                               rtol=1e-4, atol=1e-6)


def test_outlier_recent_tokens_are_not_clamped(monkeypatch):
    # Prompt-recent tokens keep their block-0 signature when the first slide makes them retrieval
    # candidates, so their c must decode within half a step even when they are outliers.
    fp = _make_cache(monkeypatch, qcenter="fp")
    _prefill(fp, recent_scale=3.0)
    exact = fp.q_bias[0][:, :SEQ].clone()
    q8 = _make_cache(monkeypatch, qcenter="8bit")
    _prefill(q8, recent_scale=3.0)
    coded = q8._qbias_from_signatures(0)[:, :SEQ]
    step = (exact[:, BLOCK0].amax(dim=1, keepdim=True) - exact[:, BLOCK0].amin(dim=1, keepdim=True)) / 255.0
    assert ((coded[:, RECENT] - exact[:, RECENT]).abs() <= step * 0.5 + 1e-3).all()
