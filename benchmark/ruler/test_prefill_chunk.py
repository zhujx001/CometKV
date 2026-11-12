import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_llm_module(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / ".." / "model_hub" / "LLM.py"
    module_path = module_path.resolve()

    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.activation = types.SimpleNamespace(silu_and_mul=lambda x, out: out.copy_(x[..., : out.shape[-1]]))
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)

    fake_termcolor = types.ModuleType("termcolor")
    fake_termcolor.colored = lambda text, _color: text
    monkeypatch.setitem(sys.modules, "termcolor", fake_termcolor)

    spec = importlib.util.spec_from_file_location("llm_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layer_prefill_splits_mlp_work_into_32768_token_chunks(monkeypatch):
    llm_module = load_llm_module(monkeypatch)
    model = llm_module.LLM.__new__(llm_module.LLM)

    class FakeLayer:
        input_layernorm_variance_epsilon = 1e-5
        input_layernorm_weight = None
        post_attention_layernorm_variance_epsilon = 1e-5
        post_attention_layernorm_weight = None

    class FakeCache:
        def prefill_update_kv_cache(self, query_states, key_states, value_states, _layer_idx, _start_bdx):
            return key_states, value_states

        def sync(self, _layer_idx, _start_bdx):
            return None

    model.layers = [FakeLayer()]
    model.num_heads = 1
    model.num_key_value_heads = 1
    model.head_dim = 1
    model.kv_cache = FakeCache()
    model.layernorm = lambda hidden_states, _eps, _weight: hidden_states
    model.wqkv = lambda hidden_states, _layer: (
        torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1, dtype=hidden_states.dtype),
        torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1, dtype=hidden_states.dtype),
        torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1, dtype=hidden_states.dtype),
    )
    model.position_embedd = lambda query_states, key_states: (query_states, key_states)
    model.prefill_attention = lambda query_states, _key_states, _value_states, _layer_idx: torch.zeros(
        query_states.shape[0],
        query_states.shape[1],
        4,
        dtype=query_states.dtype,
    )
    model.wo = lambda _attn_out, _layer, bsz, seq_len, dim: torch.zeros(bsz, seq_len, dim)

    seen_chunks = []

    def fake_mlp(hidden_states, _layer):
        seen_chunks.append(hidden_states.shape[1])
        return hidden_states

    model.mlp = fake_mlp

    hidden_states = torch.zeros(1, 70000, 4)
    model.layer_prefill(0, 0, hidden_states)

    assert seen_chunks == [32768, 32768, 4464]
