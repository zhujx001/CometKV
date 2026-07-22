import types

import pytest
import torch

from model_hub.llama import LlamaModel
from model_hub import llama as llama_module
from model_hub.qwen import QwenModel
from model_hub import qwen as qwen_module


@pytest.mark.parametrize(
    ("model_cls", "module"),
    [
        (LlamaModel, llama_module),
        (QwenModel, qwen_module),
    ],
)
def test_single_batch_decode_position_ids_reuse_slice_storage(monkeypatch, model_cls, module):
    model = model_cls.__new__(model_cls)
    model.position_ids = torch.arange(32, dtype=torch.int32)
    model.kv_cache = types.SimpleNamespace(context=5)
    model.head_dim = 128
    model.cos_sin_cache = torch.empty((32, 128))
    query_states = torch.zeros((1, 1, 512), dtype=torch.float32)
    key_states = torch.zeros((1, 1, 256), dtype=torch.float32)
    captured = {}

    def fake_apply_rope(position_ids, query, key, *_args, **_kwargs):
        captured["position_ids"] = position_ids

    monkeypatch.setattr(module.flashinfer.rope, "apply_rope_with_cos_sin_cache_inplace", fake_apply_rope)

    model.position_embedd(query_states, key_states)

    position_ids = captured["position_ids"]
    assert tuple(position_ids.shape) == (1, 1)
    assert position_ids.is_contiguous()
    assert position_ids.data_ptr() == model.position_ids[5:6].data_ptr()


@pytest.mark.parametrize("module", [llama_module, qwen_module])
def test_position_id_cache_is_int32_for_flashinfer_rope(module):
    position_ids = module._make_position_ids(32, "cpu")

    assert position_ids.dtype == torch.int32
    assert torch.equal(position_ids, torch.arange(32, dtype=torch.int32))
