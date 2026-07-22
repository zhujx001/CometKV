import importlib.util
from pathlib import Path

import pytest
import torch


def load_module(filename):
    module_path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location("bench_cometkv_fwe_sweep", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bench_module():
    return load_module("bench_cometkv_fwe_sweep.py")


def test_parse_token_lengths_accepts_k_suffix_and_plain_ints():
    bench = load_bench_module()

    assert bench.parse_token_lengths("32k,65536,96K") == [32768, 65536, 98304]


def test_decode_metrics_ignore_first_step_for_tpot_and_tokens_per_second():
    bench = load_bench_module()

    metrics = bench.compute_decode_metrics(
        ignored_decode_s=0.100,
        timed_decode_s=0.075,
        ignored_decode_steps=1,
        timed_decode_steps=3,
        batch_size=2,
    )

    assert metrics["raw_decode_s"] == pytest.approx(0.175)
    assert metrics["raw_tpot_ms"] == pytest.approx(43.75)
    assert metrics["tpot_ms"] == pytest.approx(25.0)
    assert metrics["tokens_per_second"] == pytest.approx(80.0)


def test_decode_metrics_from_profile_stats_use_timed_section():
    bench = load_bench_module()

    metrics = bench.decode_metrics_from_profile_stats(
        {
            "raw_decode_s": 0.200,
            "ignored_decode_s": 0.080,
            "timed_decode_s": 0.120,
            "ignored_decode_steps": 1,
            "timed_decode_steps": 4,
        },
        batch_size=3,
    )

    assert metrics["raw_decode_s"] == pytest.approx(0.200)
    assert metrics["raw_tpot_ms"] == pytest.approx(40.0)
    assert metrics["tpot_ms"] == pytest.approx(30.0)
    assert metrics["tokens_per_second"] == pytest.approx(100.0)


def test_make_same_length_batch_repeats_without_padding():
    bench = load_bench_module()
    input_ids = torch.arange(5, dtype=torch.int64).view(1, 5)
    attention_mask = torch.ones((1, 5), dtype=torch.int64)

    batch_ids, batch_mask = bench.make_same_length_batch(input_ids, attention_mask, 3)

    assert batch_ids.shape == (3, 5)
    assert batch_mask.shape == (3, 5)
    assert torch.equal(batch_ids[2], input_ids[0])
    assert torch.equal(batch_mask, torch.ones((3, 5), dtype=torch.int64))


def test_subprocess_command_includes_single_length_and_batch(tmp_path):
    bench = load_bench_module()
    args = bench.parse_args(
        [
            "--lengths",
            "32k,64k",
            "--batch_sizes",
            "1,2",
            "--model_path",
            "/models/llama",
            "--output_dir",
            str(tmp_path),
        ]
    )

    cmd = bench.build_worker_command(
        args,
        target_context_len=32768,
        batch_size=2,
        worker_jsonl=tmp_path / "worker.jsonl",
        worker_csv=tmp_path / "worker.csv",
    )

    assert "--worker" in cmd
    assert cmd[cmd.index("--lengths") + 1] == "32768"
    assert cmd[cmd.index("--batch_sizes") + 1] == "2"
    assert cmd[cmd.index("--jsonl_name") + 1] == "worker.jsonl"


def test_latency_entrypoint_defaults_to_fwe_data_and_single_batch():
    latency = load_module("bench_cometkv_fwe_latency.py")

    argv = latency.build_sweep_argv(["--model_path", "/models/llama"])

    assert argv[argv.index("--data_path") + 1].endswith("/test_data/fwe.json")
    assert argv[argv.index("--lengths") + 1] == "32k,64k,96k"
    assert argv[argv.index("--batch_sizes") + 1] == "1"
    assert argv[-2:] == ["--model_path", "/models/llama"]


def test_batch_entrypoint_defaults_to_16k_multi_batch():
    batch = load_module("bench_cometkv_fwe_batch_throughput.py")

    argv = batch.build_sweep_argv(["--model_path", "/models/llama"])

    assert argv[argv.index("--data_path") + 1].endswith("/test_data/fwe.json")
    assert argv[argv.index("--lengths") + 1] == "16k"
    assert argv[argv.index("--batch_sizes") + 1] == "1,2,4,8,16"
    assert argv[-2:] == ["--model_path", "/models/llama"]


def test_run_scripts_expose_model_and_batch_overrides():
    repo_root = Path(__file__).resolve().parents[2]
    latency_script = repo_root / "scripts" / "run_fwe_latency.sh"
    batch_script = repo_root / "scripts" / "run_fwe_batch_throughput.sh"

    latency_text = latency_script.read_text()
    batch_text = batch_script.read_text()

    assert "/test_data/fwe.json" in latency_text
    assert "MODEL_PATH=" in latency_text
    assert "BATCH_SIZE=" in latency_text
    assert "bench_cometkv_fwe_latency.py" in latency_text
    assert "/test_data/fwe.json" in batch_text
    assert "MODEL_PATH=" in batch_text
    assert "BATCH_SIZES=" in batch_text
    assert "bench_cometkv_fwe_batch_throughput.py" in batch_text
