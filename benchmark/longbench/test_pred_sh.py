import os
import subprocess
from pathlib import Path



def run_pred_sh(attn_type, extra_env=None):
    script = Path(__file__).resolve().with_name('pred.sh')
    env = {**os.environ, 'LONG_BENCH_DRY_RUN': '1'}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ['bash', str(script), 'llama-3.1-8b', 'qasper', attn_type, 'bf16', '0.1'],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )



def test_pred_sh_uses_single_gpu_device_for_cometkv():
    result = run_pred_sh('CometKV')

    assert result.returncode == 0
    assert '--device cuda:0' in result.stdout



def test_pred_sh_rejects_hashcluster():
    result = run_pred_sh('hashcluster')

    assert result.returncode != 0
    assert 'Unsupported attention type: hashcluster' in result.stderr



def test_pred_sh_allows_device_override():
    result = run_pred_sh('CometKV', {'DEVICE': 'cuda:3'})

    assert result.returncode == 0
    assert '--device cuda:3' in result.stdout


def test_pred_sh_can_exclude_cometkv_preserved_tokens_from_budget():
    result = run_pred_sh('CometKV', {'COMETKV_EXCLUDE_PRESERVED_FROM_BUDGET': '1'})

    assert result.returncode == 0
    assert '--cometkv_exclude_preserved_from_budget' in result.stdout


def test_pred_sh_can_count_cometkv_preserved_tokens_in_budget():
    result = run_pred_sh('CometKV', {'COMETKV_INCLUDE_PRESERVED_IN_BUDGET': '1'})

    assert result.returncode == 0
    assert '--cometkv_include_preserved_in_budget' in result.stdout


def test_pred_sh_rejects_removed_legacy_placeholder():
    script = Path(__file__).resolve().with_name('pred.sh')
    result = subprocess.run(
        ['bash', str(script), 'llama-3.1-8b', 'qasper', 'CometKV', 'bf16', '0.1', '0.232'],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env={**os.environ, 'LONG_BENCH_DRY_RUN': '1'},
        check=False,
    )

    assert result.returncode != 0
    assert '<legacy_placeholder>' not in result.stdout + result.stderr
