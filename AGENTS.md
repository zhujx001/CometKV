# Repository Guidelines

## Project Structure & Module Organization

CometKV is a Python/CUDA implementation for accelerating long-context LLM decoding.

- `model_hub/`, `attn_hub/`, and `cache_hub/` contain model adapters, attention backends, and KV-cache implementations.
- `library/cometkv/cometkv/src/` contains CUDA kernels; `library/cometkv/setup.py` builds the extensions.
- `benchmark/longbench/` and `benchmark/ruler/` contain evaluation pipelines; `scripts/` provides launch wrappers and data preparation.
- Tests live in `test/`, `library/cometkv/test/`, and beside benchmark scripts.
- `config/paths.json` defines model/data locations. Bundled inputs live in `test_data/` and `throughput_eval/test_data/`; generated outputs belong under `results/`.

## Build, Test, and Development Commands

Use Python 3.10 and a CUDA 12.x toolkit compatible with PyTorch. Run commands from the repository root unless indicated.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-cu128.txt
(cd library/cometkv && python -m pip install .)
export PYTHONPATH="$PWD:$PWD/library/cometkv${PYTHONPATH:+:$PYTHONPATH}"
```

Set `CUDA_HOME` to your toolkit before building. Rebuild after kernel changes.

- `python simple_test.py --model_name /path/to/model --attn_type CometKV --use_cuda_graph`: run an inference smoke check.
- `DATA_ROOT=/path/to/data bash scripts/prepare_data.sh`: prepare benchmark datasets.
- `DRY_RUN=1 bash scripts/run_longbench.sh`: inspect benchmark commands.
- `bash scripts/run_ruler.sh`: evaluate RULER using prepared data.

## Coding Style & Naming Conventions

Use four-space indentation in Python and follow surrounding CUDA/C++ formatting (C++17). Use `snake_case` for functions and variables, descriptive `test_*` names, and uppercase environment settings such as `COMETKV_SAMPLE_FRAC`. Preserve established public backend names. No formatter or linter configuration is checked in; keep changes consistent with nearby code.

## Testing Guidelines

Tests use pytest; no coverage percentage is enforced. Run lightweight regressions with:

```bash
python -m pytest -q test/test_*.py benchmark/longbench benchmark/ruler scripts/test_run_cometkv_benchmarks.py
```

After building on a compatible GPU, run:

```bash
python -m pytest -q library/cometkv/test/test_cometkv_*.py
```

Add focused regressions for behavior changes. CometKV batches require equally sized, unpadded prompts.

## Commit & Pull Request Guidelines

History is limited; recent documentation commits use `docs: <description>`. Use concise, descriptive subjects. PRs should explain the change, link relevant issues, and report validation commands. For performance changes, include GPU/model details, context length, retrieval budget, and before/after measurements.

## Configuration & Artifacts

Use `COMETKV_PATH_CONFIG` for machine-specific paths. Keep credentials, model weights, downloaded datasets, and generated benchmark outputs out of commits.
