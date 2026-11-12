import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_model_path


def write_shared_cache(base_dir, task, max_seq_length, metadata):
    task_dir = base_dir / "synthetic" / str(max_seq_length) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "validation.jsonl").write_text('{"index": 0}\n', encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return task_dir


def make_fake_python(bin_dir):
    python_path = bin_dir / "python"
    python_path.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"PYTHON_CALL:$*\"\n"
        "echo \"PYTHON_CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-}\"\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)


def run_ruler(script, tmp_path, task="vt", attn_type="CometKV", extra_env=None):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    make_fake_python(fake_bin)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RULER_DRY_RUN": "1",
        "DATA_CACHE_DIR": str(tmp_path / "cache"),
        "ROOT_DIR": str(tmp_path / "results"),
    }
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["bash", str(script), "llama-3.1-8b", "full", attn_type, "4096", task, "bf16", "0.018"],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def expected_metadata():
    return {
        "benchmark": "synthetic",
        "task": "vt",
        "subset": "validation",
        "num_samples": 50,
        "max_seq_length": 4096,
        "tokenizer_path": get_model_path("llama-3.1-8b"),
        "tokenizer_type": "hf",
        "model_template_type": "meta-chat",
    }


def test_ruler_run_skips_prepare_when_shared_cache_matches(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")
    cache_dir = tmp_path / "cache"
    task_dir = write_shared_cache(cache_dir, "vt", 4096, expected_metadata())

    result = run_ruler(script, tmp_path)

    assert result.returncode == 0
    assert f"CACHE_HIT: {task_dir}" in result.stdout
    assert "SKIP_PREPARE" in result.stdout
    assert "data/prepare.py" not in result.stdout


def test_ruler_run_regenerates_when_metadata_mismatches(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")
    cache_dir = tmp_path / "cache"
    write_shared_cache(
        cache_dir,
        "vt",
        4096,
        dict(expected_metadata(), num_samples=64),
    )

    result = run_ruler(script, tmp_path)

    assert result.returncode == 0
    assert "CACHE_MISS" in result.stdout
    assert "data/prepare.py" in result.stdout


def test_ruler_run_dry_run_exposes_gpu1_and_shared_paths(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")
    cache_dir = tmp_path / "cache"
    task_dir = write_shared_cache(cache_dir, "vt", 4096, expected_metadata())
    results_dir = tmp_path / "results"

    result = run_ruler(script, tmp_path)

    assert result.returncode == 0
    assert "CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert str(task_dir) in result.stdout
    assert f"{results_dir}/{get_model_path('llama-3.1-8b')}/synthetic/4096/CometKV/pred" in result.stdout


def test_ruler_run_rejects_removed_attn_type(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")

    result = run_ruler(script, tmp_path, attn_type="hashcluster")

    assert result.returncode != 0
    assert "Unsupported attention type: hashcluster" in result.stderr


def test_ruler_run_rejects_removed_reserved_budget_arg(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")

    result = subprocess.run(
        ["bash", str(script), "llama-3.1-8b", "full", "CometKV", "4096", "vt", "bf16", "0.018", "0.232"],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env={**os.environ, "RULER_DRY_RUN": "1"},
        check=False,
    )

    assert result.returncode != 0
    assert "reserved" not in result.stdout + result.stderr


def test_ruler_run_all_expands_all_synthetic_tasks(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")
    cache_dir = tmp_path / "cache"
    for task in (
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_1",
        "qa_2",
    ):
        write_shared_cache(cache_dir, task, 4096, dict(expected_metadata(), task=task))

    result = run_ruler(script, tmp_path, task="ALL")

    assert result.returncode == 0
    assert "RUN niah_single_1" in result.stdout
    assert "RUN qa_2" in result.stdout
    assert result.stdout.count("pred/call_api.py") == 13


def test_ruler_run_only_cleans_current_task_prediction_files(tmp_path):
    script = Path(__file__).resolve().with_name("ruler_run.sh")
    cache_dir = tmp_path / "cache"
    write_shared_cache(cache_dir, "qa_2", 4096, dict(expected_metadata(), task="qa_2"))

    result = run_ruler(script, tmp_path, task="qa_2")

    assert result.returncode == 0
    assert "RESET_PRED_DIR" not in result.stdout
    assert "CLEAN_PRED_FILE:" in result.stdout
    assert "qa_2.jsonl" in result.stdout
    assert "qa_2-*.jsonl" in result.stdout
