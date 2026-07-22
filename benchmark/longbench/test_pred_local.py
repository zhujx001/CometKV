import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


def load_pred_module(monkeypatch):
    module_path = Path(__file__).resolve().with_name("pred.py")
    monkeypatch.chdir(module_path.parent)
    fake_model_hub = types.ModuleType("model_hub")

    def add_model_args(parser):
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--dtype", type=str, default="bf16")
        parser.add_argument("--model_name", type=str, default="unused")
        return parser

    fake_model_hub.LlamaModel = object
    fake_model_hub.QwenModel = object
    fake_model_hub.MistralModel = object
    fake_model_hub.add_model_args = add_model_args
    monkeypatch.setitem(sys.modules, "model_hub", fake_model_hub)
    spec = importlib.util.spec_from_file_location("longbench_pred", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_local_defaults(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args(["--task", "qasper"])

    assert args.model_path == pred.DEFAULT_MODEL_PATH
    assert args.data_dir == pred.DEFAULT_DATA_DIR


def test_load_local_dataset_reads_jsonl_and_e_suffix(monkeypatch, tmp_path):
    pred = load_pred_module(monkeypatch)
    sample = {
        "input": "prompt",
        "answers": ["answer"],
        "all_classes": None,
        "length": 8,
    }

    for name in ("demo.jsonl", "demo_e.jsonl"):
        with (tmp_path / name).open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    assert pred.load_local_dataset(str(tmp_path), "demo", False) == [sample]
    assert pred.load_local_dataset(str(tmp_path), "demo", True) == [sample]


def test_parse_args_allows_model_and_data_overrides(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--model", "qwen2.5-7b",
        "--model_path", "/tmp/qwen",
        "--data_dir", "/tmp/longbench",
    ])

    assert args.model == "qwen2.5-7b"
    assert args.model_path == "/tmp/qwen"
    assert args.data_dir == "/tmp/longbench"


def test_resolve_config_model_name_uses_known_alias_for_local_override(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config_model_name = pred.resolve_config_model_name(
        "llama-3.1-8b",
        "/models/Llama-3.1-8B-Instruct",
    )

    assert config_model_name == "Llama-3.1-8B-Instruct"


def test_parse_args_accepts_mistral_default_path(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--model", "mistral-7b-instruct-v0.2",
    ])

    assert args.model == "mistral-7b-instruct-v0.2"
    assert args.model_path == pred.get_model_path("mistral-7b-instruct-v0.2")


def test_parse_args_accepts_cometkv_attn_type(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--attn_type", "CometKV",
    ])

    assert args.attn_type == "CometKV"
    assert not hasattr(args, "cometkv_token_cache_lru")
    assert not hasattr(args, "cometkv_prefill_sync_mode")
    assert not hasattr(args, "cometkv_prefill_async_window")
    assert args.cometkv_min_retrieval_topk == 16
    assert args.cometkv_exclude_preserved_from_budget is True
    assert not hasattr(args, "hashcluster_method")
    assert not hasattr(args, "estimation_budget")
    assert not hasattr(args, "cache_ratio")
    assert not hasattr(args, "gpu_only")


def test_parse_args_rejects_hashcluster_attn_type(monkeypatch):
    pred = load_pred_module(monkeypatch)

    with pytest.raises(SystemExit):
        pred.parse_args([
            "--task", "qasper",
            "--attn_type", "hashcluster",
        ])


def test_parse_args_accepts_generation_timing_controls(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--profile_timing",
        "--prefill_bsz", "4",
    ])

    assert args.profile_timing is True
    assert args.prefill_bsz == 4


def test_parse_args_rejects_removed_cometkv_lru(monkeypatch):
    pred = load_pred_module(monkeypatch)

    with pytest.raises(SystemExit):
        pred.parse_args([
            "--task", "qasper",
            "--attn_type", "CometKV",
            "--cometkv_token_cache_lru",
        ])


def test_parse_args_accepts_cometkv_static_pattern_overrides(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--attn_type", "CometKV",
        "--cometkv_static_pattern_start", "4",
        "--cometkv_static_pattern_end", "32",
    ])

    assert args.cometkv_static_pattern_start == 4
    assert args.cometkv_static_pattern_end == 32


def test_parse_args_accepts_cometkv_min_retrieval_topk(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--attn_type", "CometKV",
        "--cometkv_min_retrieval_topk", "0",
    ])

    assert args.cometkv_min_retrieval_topk == 0


def test_parse_args_accepts_cometkv_budget_scope_flag(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--attn_type", "CometKV",
        "--cometkv_exclude_preserved_from_budget",
    ])

    assert args.cometkv_exclude_preserved_from_budget is True


def test_parse_args_accepts_cometkv_budget_scope_opt_out(monkeypatch):
    pred = load_pred_module(monkeypatch)

    args = pred.parse_args([
        "--task", "qasper",
        "--attn_type", "CometKV",
        "--cometkv_include_preserved_in_budget",
    ])

    assert args.cometkv_exclude_preserved_from_budget is False


def test_generate_config_builds_cometkv_defaults(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
    )

    assert "CometKV" in config
    assert config["CometKV"]["sig_bits"] == 128
    assert config["CometKV"]["sig_mode"] == "random_orth"
    assert config["CometKV"]["static_pattern_start"] == 4
    assert config["CometKV"]["sig_chunk_size"] == 131072
    assert config["CometKV"]["sig_token_cache_size"] == 1024
    assert config["CometKV"]["sig_min_retrieval_topk"] == 16
    assert config["CometKV"]["exclude_preserved_from_budget"] is True
    assert "sig_union3_prefetch_ratio" not in config["CometKV"]
    assert "sig_cache_admission_distance_slack" not in config["CometKV"]
    assert "sig_cache_admission_topk_ratio" not in config["CometKV"]
    assert "sig_token_cache_lru" not in config["CometKV"]
    assert "sig_token_cache_protected_lru" not in config["CometKV"]
    assert "sig_token_cache_protect_window" not in config["CometKV"]
    assert "sig_prefill_sync_mode" not in config["CometKV"]
    assert "sig_prefill_async_window" not in config["CometKV"]
    assert "sig_concat_flash_attention" not in config["CometKV"]
    assert "sig_static_fixed_prompt_local" not in config["CometKV"]
    assert "sig_gpu_recent_tokens" not in config["CometKV"]
    assert "sig_gpu_index" not in config["CometKV"]
    assert "sig_decode_backend" not in config["CometKV"]
    assert "sig_use_block_rep" not in config["CometKV"]
    assert "sig_retrieval_scope" not in config["CometKV"]
    assert "sig_flash_threshold" not in config["CometKV"]
    assert "sig_page_size" not in config["CometKV"]
    assert "spill_staging_tokens" not in config["CometKV"]
    assert "use_pred_q" not in config["CometKV"]


def test_generate_config_applies_cometkv_static_pattern_overrides(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
        cometkv_static_pattern_start=4,
        cometkv_static_pattern_end=32,
    )

    assert config["CometKV"]["static_pattern_start"] == 4
    assert config["CometKV"]["static_pattern_end"] == 32


def test_generate_config_sets_cometkv_min_retrieval_topk(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
        cometkv_min_retrieval_topk=0,
    )

    assert config["CometKV"]["sig_min_retrieval_topk"] == 0


def test_generate_config_sets_cometkv_budget_scope(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
        cometkv_exclude_preserved_from_budget=True,
    )

    assert config["CometKV"]["exclude_preserved_from_budget"] is True


def test_generate_config_can_count_preserved_tokens_in_budget(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
        cometkv_exclude_preserved_from_budget=False,
    )

    assert config["CometKV"]["exclude_preserved_from_budget"] is False


def test_generate_config_sets_cometkv_sig_bits(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
        sig_bits=64,
    )

    assert config["CometKV"]["sig_bits"] == 64


def test_generate_config_sets_cometkv_token_cache_size(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.DEFAULT_MODEL_PATH,
        4096,
        "CometKV",
        cometkv_token_cache_size=20480,
    )

    assert config["CometKV"]["sig_token_cache_size"] == 20480


def test_generate_config_rejects_removed_hashcluster_attn(monkeypatch):
    pred = load_pred_module(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported attention type"):
        pred.generate_config(
            pred.DEFAULT_MODEL_PATH,
            32768,
            "hashcluster",
        )


def test_generate_config_builds_mistral_cometkv_config(monkeypatch):
    pred = load_pred_module(monkeypatch)

    config = pred.generate_config(
        pred.get_model_path("mistral-7b-instruct-v0.2"),
        4096,
        "CometKV",
    )

    assert "CometKV" in config
    assert config["CometKV"]["retrieval_budget"] == 0.02


def test_load_model_dispatches_mistral(monkeypatch):
    pred = load_pred_module(monkeypatch)
    created = {}

    class FakeMistralModel:
        def __init__(self, model_path, max_length, dtype, device_map):
            created["args"] = (model_path, max_length, dtype, device_map)
            self.tokenizer = types.SimpleNamespace(eos_token="<eos>")

    fake_model_hub = types.ModuleType("model_hub")
    fake_model_hub.LlamaModel = object
    fake_model_hub.QwenModel = object
    fake_model_hub.MistralModel = FakeMistralModel
    monkeypatch.setitem(sys.modules, "model_hub", fake_model_hub)

    llm = pred.load_model(pred.get_model_path("mistral-7b-instruct-v0.2"), 130000, torch.bfloat16, "auto")

    assert isinstance(llm, FakeMistralModel)
    assert created["args"] == (
        pred.get_model_path("mistral-7b-instruct-v0.2"),
        130000,
        torch.bfloat16,
        "auto",
    )
    assert llm.tokenizer.pad_token == "<eos>"
    assert llm.tokenizer.padding_side == "left"


def test_pred_prepends_local_kernel_library_path(monkeypatch):
    pred = load_pred_module(monkeypatch)

    assert sys.path[0] == pred.KERNEL_LIB


class FakeBatch(dict):
    def __init__(self, input_ids):
        tensor = torch.tensor([input_ids], dtype=torch.long)
        super().__init__(input_ids=tensor, attention_mask=torch.ones_like(tensor))
        self.input_ids = tensor
        self.attention_mask = torch.ones_like(tensor)

    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.applied_messages = None
        self.seen_prompts = []
        self.seen_add_special_tokens = []
        self.batch_decode_kwargs = None
        self.eos_token = "<eos>"
        self.eos_token_id = 9

    def __call__(self, text, truncation=False, return_tensors=None, padding=False, add_special_tokens=True):
        if isinstance(text, list):
            text = text[0]
        self.seen_prompts.append(text)
        self.seen_add_special_tokens.append(add_special_tokens)
        token_ids = [ord(char) for char in text]
        return FakeBatch(token_ids)

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)

    def batch_decode(self, outputs, **kwargs):
        self.batch_decode_kwargs = kwargs
        return ["".join(chr(token_id) for token_id in output) for output in outputs]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.applied_messages = messages
        return f"CHAT::{messages[1]['content']}"


def test_prepare_prompt_applies_chat_template_for_llama(monkeypatch):
    pred = load_pred_module(monkeypatch)
    tokenizer = FakeTokenizer()

    prompt = pred.prepare_prompt(
        tokenizer=tokenizer,
        prompt_format="Question: {input}",
        json_obj={"input": "hello"},
        dataset="qasper",
        model_name="llama-3.1-8b",
        # prompt budget = max_length - max_new_tokens - CHAT_TEMPLATE_OVERHEAD_ESTIMATE(64)
        # must stay positive.
        max_length=256,
        max_new_tokens=8,
    )

    assert prompt == "CHAT::Question: hello"
    assert tokenizer.applied_messages[1]["content"] == "Question: hello"


def test_tokenize_prompt_disables_added_special_tokens_for_chat_template(monkeypatch):
    pred = load_pred_module(monkeypatch)
    tokenizer = FakeTokenizer()

    pred.tokenize_prompt(
        tokenizer=tokenizer,
        prompt="CHAT::Question: hello",
        use_chat_template=True,
        padding=True,
    )

    assert tokenizer.seen_add_special_tokens[-1] is False


def test_prepare_prompt_skips_chat_template_for_plain_completion_tasks(monkeypatch):
    pred = load_pred_module(monkeypatch)
    tokenizer = FakeTokenizer()

    prompt = pred.prepare_prompt(
        tokenizer=tokenizer,
        prompt_format="Question: {input}",
        json_obj={"input": "hello"},
        dataset="trec",
        model_name="llama-3.1-8b",
        max_length=256,
        max_new_tokens=8,
    )

    assert prompt == "Question: hello"
    assert tokenizer.applied_messages is None


def test_prepare_prompt_truncates_to_leave_generation_budget(monkeypatch):
    pred = load_pred_module(monkeypatch)
    tokenizer = FakeTokenizer()

    prompt = pred.prepare_prompt(
        tokenizer=tokenizer,
        prompt_format="{context}",
        json_obj={"context": "abcdefghij"},
        dataset="trec",
        model_name="llama-3.1-8b",
        # prompt budget = 72 - 2 - 64 = 6 -> the 10-char context truncates to head 3 + tail 3.
        max_length=72,
        max_new_tokens=2,
    )

    assert prompt == "abchij"


def test_resolve_eos_token_ids_prefers_model_config(monkeypatch):
    pred = load_pred_module(monkeypatch)
    llm = types.SimpleNamespace(
        config=types.SimpleNamespace(eos_token_id=[128001, 128008, 128009]),
        tokenizer=types.SimpleNamespace(eos_token_id=128009),
    )

    assert pred.resolve_eos_token_ids(llm) == [128001, 128008, 128009]


def test_get_pred_uses_prepared_prompt_and_eos_stop(monkeypatch, tmp_path):
    pred = load_pred_module(monkeypatch)
    tokenizer = FakeTokenizer()

    class FakeLLM:
        def __init__(self):
            self.tokenizer = tokenizer
            self.layers = [types.SimpleNamespace(device="cpu")]
            self.config = types.SimpleNamespace(eos_token_id=[7, 8])
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return [[ord("o"), ord("k")]]

    llm = FakeLLM()
    monkeypatch.setattr(pred, "attn_type", "Full_Flash_Attn", raising=False)
    monkeypatch.setattr(pred, "generate_config", lambda *args, **kwargs: {"Full_Flash_Attn": {}})
    monkeypatch.setattr(pred.torch.cuda, "empty_cache", lambda: None)

    sample = {
        "context": "ctx",
        "input": "question",
        "answers": ["answer"],
        "all_classes": None,
        "length": 3,
    }
    out_path = tmp_path / "pred.jsonl"

    pred.get_pred(
        llm=llm,
        data=[sample],
        max_new_tokens=4,
        prompt_format="Context: {context}\nQuestion: {input}",
        model_name="llama-3.1-8b",
        model_path=pred.DEFAULT_MODEL_PATH,
        max_length=256,
        dataset="qasper",
        out_path=str(out_path),
        args=types.SimpleNamespace(retrieval_budget=None, prefill_bsz=3, profile_timing=True),
    )

    # truncate_prompt tokenizes the RAW prompt first (budget check); the chat-wrapped
    # prompt is the LAST tokenizer call — the one actually fed to generation.
    assert tokenizer.seen_prompts[-1].startswith("CHAT::")
    assert tokenizer.batch_decode_kwargs == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    assert llm.generate_kwargs["ignore_eos"] is False
    assert llm.generate_kwargs["eos_token_ids"] == [7, 8]
    assert llm.generate_kwargs["prefill_bsz"] == 3
    assert llm.generate_kwargs["profile_timing"] is True
