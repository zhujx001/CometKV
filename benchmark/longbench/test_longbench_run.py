import subprocess
import os
from pathlib import Path



def test_longbench_run_accepts_direct_task_in_dry_run():
    script = Path(__file__).resolve().with_name('longbench_run.sh')
    result = subprocess.run(
        ['bash', str(script), 'llama-3.1-8b', 'CometKV', '0.1', 'bf16', 'qasper'],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env={**os.environ, 'LONG_BENCH_DRY_RUN': '1'},
        check=False,
    )

    assert result.returncode == 0
    assert 'Parameters: llama-3.1-8b qasper CometKV bf16 0.1' in result.stdout
    assert 'Unknown CATEGORY' not in result.stdout


def test_longbench_run_accepts_all_category_in_dry_run():
    script = Path(__file__).resolve().with_name('longbench_run.sh')
    result = subprocess.run(
        ['bash', str(script), 'llama-3.1-8b', 'CometKV', '0.1', 'bf16', 'ALL'],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env={**os.environ, 'LONG_BENCH_DRY_RUN': '1'},
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.count('Parameters:') == 20
    assert 'Parameters: llama-3.1-8b qasper CometKV bf16 0.1' in result.stdout
    assert 'Parameters: llama-3.1-8b lcc CometKV bf16 0.1' in result.stdout
    assert 'Unknown CATEGORY' not in result.stdout


def test_longbench_run_rejects_removed_legacy_placeholder():
    script = Path(__file__).resolve().with_name('longbench_run.sh')
    result = subprocess.run(
        ['bash', str(script), 'llama-3.1-8b', 'CometKV', '0.1', '0.232', 'bf16', 'qasper'],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env={**os.environ, 'LONG_BENCH_DRY_RUN': '1'},
        check=False,
    )

    assert result.returncode != 0
    assert '<legacy_placeholder>' not in result.stdout + result.stderr
