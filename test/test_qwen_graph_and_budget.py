from types import SimpleNamespace
import importlib

import pytest
import torch

from model_hub.qwen import QwenModel
from cache_hub.test_exact_topk_cache import make_cache, BSZ, HEADS, DIM, LAYERS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def test_qwen_graph_replay_reads_current_position():
    model = QwenModel.__new__(QwenModel)
    model.head_dim = 128
    model.position_ids = torch.arange(64, dtype=torch.int32, device='cuda')
    frequency = 1.0 / (10000 ** (torch.arange(0, 128, 2, device='cuda').float() / 128))
    angle = torch.outer(torch.arange(64, device='cuda').float(), frequency)
    model.cos_sin_cache = torch.cat([angle.cos(), angle.sin()], -1)
    model._cg_position_ids = torch.tensor([3, 3], dtype=torch.int32, device='cuda')
    model.kv_cache = SimpleNamespace(context=3)
    q = torch.randn(2, 1, 256, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(2, 1, 128, dtype=torch.bfloat16, device='cuda')
    model.use_cuda_graph = True
    def run():
        return model.position_embedd(q.clone(), k.clone())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_k = run()
    for pos in [7, 11, 31]:
        model._cg_position_ids.fill_(pos)
        graph.replay()
        model.use_cuda_graph = False
        model.kv_cache.context = pos
        expected_q, expected_k = run()
        torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
        torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
        model.use_cuda_graph = True


def test_topk_preserved_tokens_do_not_reduce_retrieval_quota(monkeypatch):
    cache = make_cache(total=300, budget=0.02, force_sink=4, force_recent=32)
    query = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device='cuda')
    seen = {}
    def attend(**kwargs):
        seen['count'] = kwargs['k_cache'].shape[1]
        return kwargs['q']
    monkeypatch.setattr(importlib.import_module('cache_hub.exact_topk_cache'), 'flash_attn_with_kvcache', attend)
    cache.sparse_attention(query, LAYERS - 1)
    assert seen['count'] == 6 + 4 + 32
