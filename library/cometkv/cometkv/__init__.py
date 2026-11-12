import torch
import site
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_CACHE_TAG = sys.implementation.cache_tag

_preferred_paths = []

for _build_dir in sorted((_REPO_ROOT / "build").glob(f"lib.*-{_CACHE_TAG}/cometkv")):
    _build_str = str(_build_dir)
    if _build_dir.is_dir() and _build_str not in __path__:
        _preferred_paths.append(_build_str)

for _base in [*site.getsitepackages(), site.getusersitepackages()]:
    _candidate = Path(_base) / "cometkv"
    _candidate_str = str(_candidate)
    if _candidate.is_dir() and _candidate != _PKG_DIR and _candidate_str not in __path__ and _candidate_str not in _preferred_paths:
        _preferred_paths.append(_candidate_str)

if _preferred_paths:
    __path__.extend(_preferred_paths)

for _module_name in (
    "CometKVGather",
    "CometKVGpuGather",
    "CometKVSignature",
):
    try:
        exec(f"from .{_module_name} import *")
    except ImportError:
        pass
