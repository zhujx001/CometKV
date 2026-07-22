import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


def load_prepare_module(monkeypatch):
    module_path = Path(__file__).resolve().parent / "data" / "prepare.py"
    monkeypatch.chdir(module_path.parent)
    monkeypatch.syspath_prepend(str(module_path.parent))

    fake_nltk = types.ModuleType("nltk")
    fake_nltk.data = types.SimpleNamespace(find=lambda _name: True)
    fake_nltk.download = lambda _name: True
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)

    spec = importlib.util.spec_from_file_location("ruler_prepare", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_writes_cache_metadata_after_generation(monkeypatch, tmp_path):
    prepare = load_prepare_module(monkeypatch)
    save_dir = tmp_path / "cache"

    def fake_run(command, shell, check, stdout, stderr, text):
        task_dir = save_dir / "vt"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "validation.jsonl").write_text('{"index": 0}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(prepare.subprocess, "run", fake_run)

    args = argparse.Namespace(
        save_dir=save_dir,
        benchmark="synthetic",
        task="vt",
        tokenizer_path="/tmp/tokenizer",
        tokenizer_type="hf",
        max_seq_length=4096,
        model_template_type="meta-chat",
        num_samples=50,
        remove_newline_tab=False,
        chunk_idx=0,
        chunk_amount=1,
        subset="validation",
        random_seed=42,
    )

    prepare.main(args)

    metadata = json.loads((save_dir / "vt" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["benchmark"] == "synthetic"
    assert metadata["task"] == "vt"
    assert metadata["subset"] == "validation"
    assert metadata["num_samples"] == 50
    assert metadata["max_seq_length"] == 4096
    assert metadata["tokenizer_path"] == "/tmp/tokenizer"
    assert metadata["tokenizer_type"] == "hf"
    assert metadata["model_template_type"] == "meta-chat"
