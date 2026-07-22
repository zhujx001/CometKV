import os
import subprocess
from pathlib import Path


def run_script(tmp_path, extra_env=None):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_cometkv_benchmarks.sh"
    env = {
        **os.environ,
        "LONG_BENCH_DRY_RUN": "1",
        "RULER_DRY_RUN": "1",
        "PYTHON_BIN": "python",
        "MODEL_PATH": "/models/llama",
        "LONG_BENCH_DATA_DIR": "/data/longbench",
        "RULER_DATA_ROOT": "/data/RULER",
        "RESULT_ROOT": str(tmp_path / "results"),
        "LONG_BENCH_TASKS": "qasper",
        "RULER_TASKS": "fwe",
        "RULER_CONTEXT_LENGTHS": "32768",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_script_runs_default_budget_grid_and_names_result_dirs(tmp_path):
    result = run_script(tmp_path)

    assert result.returncode == 0
    assert "sink32_recent64_budget0p05" in result.stdout
    assert "sink32_recent64_budget0p10" in result.stdout
    assert "--cometkv_static_pattern_start 32" in result.stdout
    assert "--cometkv_static_pattern_end 64" in result.stdout
    assert "--cometkv_exclude_preserved_from_budget" in result.stdout
    assert "--retrieval_budget 0.05" in result.stdout
    assert "--retrieval_budget 0.1" in result.stdout
    assert result.stdout.count("BEGIN_EXPERIMENT") == 2


def test_script_allows_sink_recent_budget_and_model_overrides(tmp_path):
    result = run_script(
        tmp_path,
        {
            "MODEL_NAME": "custom-model",
            "MODEL_PATH": "/models/custom",
            "SINK_VALUES": "16",
            "RECENT_VALUES": "128",
            "BUDGETS": "0.02",
        },
    )

    assert result.returncode == 0
    assert "sink16_recent128_budget0p02" in result.stdout
    assert "--model custom-model" in result.stdout
    assert "--model_path /models/custom" in result.stdout
    assert "--cometkv_static_pattern_start 16" in result.stdout
    assert "--cometkv_static_pattern_end 128" in result.stdout
