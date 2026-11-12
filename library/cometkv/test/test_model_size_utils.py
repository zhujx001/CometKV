import argparse
import importlib.util
from pathlib import Path

import pytest


def load_utils_module():
    module_path = Path(__file__).resolve().parents[3] / "model_hub" / "utils.py"
    spec = importlib.util.spec_from_file_location("model_hub_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = load_utils_module()


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("/models/Llama-3.1-8B-Instruct", 8),
        ("Qwen2.5-72B-Instruct", 72),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 8),
    ],
)
def test_extract_model_size_billion(model_name, expected):
    assert utils.extract_model_size_billion(model_name) == expected


def test_add_model_args_accepts_local_model_path():
    parser = argparse.ArgumentParser()
    utils.add_model_args(parser)

    args = parser.parse_args(["--model_name", "/models/Llama-3.1-8B-Instruct"])

    assert args.model_name == "/models/Llama-3.1-8B-Instruct"


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
        ("/models/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
        ("/models/Qwen2.5-7B-Instruct/", "Qwen2.5-7B-Instruct"),
    ],
)
def test_model_name_key_normalizes_repo_ids_and_local_paths(model_name, expected):
    assert utils.model_name_key(model_name) == expected
