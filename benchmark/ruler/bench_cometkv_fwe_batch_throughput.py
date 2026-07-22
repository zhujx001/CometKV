#!/usr/bin/env python3
"""Run CometKV FWE same-length batch throughput benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ruler.bench_cometkv_fwe_sweep import main as sweep_main


DEFAULT_DATA_PATH = PROJECT_ROOT / "test_data" / "fwe.json"


def build_sweep_argv(extra_args: list[str] | None = None) -> list[str]:
    return [
        "--data_path",
        str(DEFAULT_DATA_PATH),
        "--lengths",
        "16k",
        "--batch_sizes",
        "1,2,4,8,16",
        "--output_dir",
        str(PROJECT_ROOT / "benchmark" / "ruler" / "speed_results" / "batch_throughput"),
        *(extra_args or []),
    ]


def main(argv: list[str] | None = None) -> int:
    return sweep_main(build_sweep_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
