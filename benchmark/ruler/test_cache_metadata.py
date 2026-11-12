import importlib.util
import json
from pathlib import Path


def load_cache_metadata_module():
    module_path = Path(__file__).resolve().parent / "data" / "cache_metadata.py"
    spec = importlib.util.spec_from_file_location("ruler_cache_metadata", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_expected_metadata(module):
    return module.build_cache_metadata(
        benchmark="synthetic",
        task="vt",
        subset="validation",
        num_samples=50,
        max_seq_length=4096,
        tokenizer_path="/tmp/tokenizer",
        tokenizer_type="hf",
        model_template_type="meta-chat",
    )


def write_cache_files(task_dir, metadata, subset="validation", data_exists=True):
    task_dir.mkdir(parents=True, exist_ok=True)
    if data_exists:
        (task_dir / f"{subset}.jsonl").write_text('{"index": 0}\n', encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def test_cache_is_valid_when_data_and_metadata_match(tmp_path):
    cache_metadata = load_cache_metadata_module()
    task_dir = tmp_path / "synthetic" / "4096" / "vt"
    expected = build_expected_metadata(cache_metadata)
    write_cache_files(task_dir, expected)

    assert cache_metadata.cache_is_valid(task_dir, expected)


def test_cache_is_invalid_without_dataset_file(tmp_path):
    cache_metadata = load_cache_metadata_module()
    task_dir = tmp_path / "synthetic" / "4096" / "vt"
    expected = build_expected_metadata(cache_metadata)
    write_cache_files(task_dir, expected, data_exists=False)

    assert not cache_metadata.cache_is_valid(task_dir, expected)


def test_cache_is_invalid_when_metadata_differs(tmp_path):
    cache_metadata = load_cache_metadata_module()
    task_dir = tmp_path / "synthetic" / "4096" / "vt"
    expected = build_expected_metadata(cache_metadata)
    mismatched = dict(expected, num_samples=64)
    write_cache_files(task_dir, mismatched)

    assert not cache_metadata.cache_is_valid(task_dir, expected)


def test_cache_is_invalid_when_metadata_is_not_json(tmp_path):
    cache_metadata = load_cache_metadata_module()
    task_dir = tmp_path / "synthetic" / "4096" / "vt"
    expected = build_expected_metadata(cache_metadata)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "validation.jsonl").write_text('{"index": 0}\n', encoding="utf-8")
    (task_dir / "metadata.json").write_text("{not-json", encoding="utf-8")

    assert not cache_metadata.cache_is_valid(task_dir, expected)
