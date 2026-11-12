"""Generated CometKV configs carry the tail accounting, sampling and token-cache settings."""

import ast
from pathlib import Path

import pytest

from config import generate_config

MODEL = "Llama-3.1-8B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("COMETKV_TAIL_IN_BUDGET", "COMETKV_SAMPLE_MIN_CONTEXT", "COMETKV_SAMPLE_SIZE",
                 "COMETKV_SAMPLE_FRAC", "COMETKV_SAMPLE_MODE", "COMETKV_SAMPLE_MAX_M", "COMETKV_SAMPLE_MIN_M",
                 "COMETKV_TOKEN_CACHE_MULT", "COMETKV_TOKEN_CACHE_ROUND"):
        monkeypatch.delenv(name, raising=False)


def cometkv(**kwargs):
    return generate_config(MODEL, 32768, "CometKV", **kwargs)["CometKV"]


PAPER = dict(tail_in_budget=True, sample_min_context=32768, sample_mode="fraction", sample_frac=0.25,
             sample_size=256, sample_max_m=256, sample_min_m=64, token_cache_mult=2.0, token_cache_round=8)


def test_defaults_follow_the_paper():
    config = cometkv()
    assert {key: config[key] for key in PAPER} == PAPER


def test_size_mode_environment_sets_a_fixed_count(monkeypatch):
    monkeypatch.setenv("COMETKV_SAMPLE_MODE", "size")
    monkeypatch.setenv("COMETKV_SAMPLE_SIZE", "64")
    config = cometkv()
    assert (config["sample_mode"], config["sample_size"]) == ("size", 64)


def test_sample_size_without_a_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("COMETKV_SAMPLE_SIZE", "64")
    with pytest.raises(ValueError, match="COMETKV_SAMPLE_MODE"):
        cometkv()


def test_arguments_select_the_legacy_accounting_and_threshold():
    config = cometkv(cometkv_tail_in_budget=False, cometkv_sample_min_context=0)
    assert config["tail_in_budget"] is False
    assert config["sample_min_context"] == 0


@pytest.mark.parametrize("flag,expected", [("0", False), ("1", True)])
def test_environment_overrides_arguments(monkeypatch, flag, expected):
    monkeypatch.setenv("COMETKV_TAIL_IN_BUDGET", flag)
    monkeypatch.setenv("COMETKV_SAMPLE_MIN_CONTEXT", "65536")
    config = cometkv(cometkv_tail_in_budget=not expected, cometkv_sample_min_context=0)
    assert config["tail_in_budget"] is expected
    assert config["sample_min_context"] == 65536


@pytest.mark.parametrize("source", ["model_hub/llama.py", "model_hub/qwen.py"])
def test_models_forward_both_settings_to_the_cache(source):
    tree = ast.parse((ROOT / source).read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "cometkv_cache"]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {"tail_in_budget", "sample_min_context", "sample_mode", "sample_frac", "sample_size",
            "sample_max_m", "sample_min_m", "token_cache_mult", "token_cache_round"} <= keywords
