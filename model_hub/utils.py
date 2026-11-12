import re

from config import known_model_paths


_MODEL_SIZE_PATTERN = re.compile(r"(\d+)[B]")
_PATH_SPLIT_PATTERN = re.compile(r"[\\/]+")
_KNOWN_MODEL_IDS = tuple(known_model_paths())


def add_model_args(parser):
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, set to `auto` to split model across all available GPUs")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type")
    parser.add_argument(
        "--model_name",
        type=str,
        default=_KNOWN_MODEL_IDS[0],
        help="Model repo id or local model path. Known tested ids: " + ", ".join(_KNOWN_MODEL_IDS),
    )
    return parser


def model_name_key(model_name: str) -> str:
    normalized = model_name.rstrip("/\\")
    if not normalized:
        return model_name
    return _PATH_SPLIT_PATTERN.split(normalized)[-1]


def extract_model_size_billion(model_name: str) -> int:
    match = _MODEL_SIZE_PATTERN.search(model_name)
    if match is None:
        raise ValueError(f"Cannot infer model size from: {model_name}")
    return int(match.group(1))
