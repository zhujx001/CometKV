import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH_CONFIG = PROJECT_ROOT / "config" / "paths.json"
PATH_CONFIG_ENV = "COMETKV_PATH_CONFIG"


def _expand_env(value):
    if not isinstance(value, str):
        return value
    return os.path.expandvars(os.path.expanduser(value))


def _expand_path(value):
    expanded = _expand_env(value)
    if isinstance(expanded, str) and expanded and not os.path.isabs(expanded):
        return str((PROJECT_ROOT / expanded).resolve())
    return expanded


def _expand_model_path(value):
    expanded = _expand_env(value)
    if isinstance(expanded, str) and expanded.startswith((".", "..")) and not os.path.isabs(expanded):
        return str((PROJECT_ROOT / expanded).resolve())
    return expanded


def _expand_for_key(key, value):
    if isinstance(value, list):
        return [_expand_path(item) for item in value]
    if key.endswith(".extra_task_files"):
        return _expand_env(value)
    if key.startswith(("datasets.", "runtime.")):
        return _expand_path(value)
    return _expand_env(value)


def load_path_config(config_path=None):
    path = Path(config_path or os.environ.get(PATH_CONFIG_ENV, DEFAULT_PATH_CONFIG))
    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def get_by_key(key, config_path=None):
    value = load_path_config(config_path)
    for part in key.split("."):
        value = value[part]
    return _expand_for_key(key, value)


def _model_entries(config):
    return config.get("models", {})


def resolve_model_config(model_name, config_path=None):
    config = load_path_config(config_path)
    models = _model_entries(config)
    if model_name in models:
        model = dict(models[model_name])
        model["name"] = model_name
        model["path"] = _expand_model_path(model["path"])
        model["tokenizer_path"] = _expand_model_path(model.get("tokenizer_path", model["path"]))
        return model

    for name, model_config in models.items():
        aliases = set(model_config.get("aliases", []))
        path = _expand_model_path(model_config["path"])
        path_name = Path(path.rstrip("/")).name
        if model_name in aliases or model_name == path or model_name == path_name:
            model = dict(model_config)
            model["name"] = name
            model["path"] = path
            model["tokenizer_path"] = _expand_model_path(model.get("tokenizer_path", path))
            return model

    raise KeyError(f"Unknown model in {PATH_CONFIG_ENV or DEFAULT_PATH_CONFIG}: {model_name}")


def get_model_path(model_name, config_path=None):
    return resolve_model_config(model_name, config_path)["path"]


def known_model_paths(config_path=None):
    return [resolve_model_config(name, config_path)["path"] for name in _model_entries(load_path_config(config_path))]


def known_model_names(config_path=None):
    return list(_model_entries(load_path_config(config_path)).keys())


def _cmd_model_select(args):
    model = resolve_model_config(args.model, args.config)
    print(
        ":".join(
            [
                model["path"],
                model.get("template_type", "meta-chat"),
                model.get("framework", "hf"),
                model.get("tokenizer_path", model["path"]),
                model.get("tokenizer_type", "hf"),
            ]
        )
    )


def _cmd_model_path(args):
    print(get_model_path(args.model, args.config))


def _cmd_model_names(args):
    print(" ".join(known_model_names(args.config)))


def _cmd_get(args):
    value = get_by_key(args.key, args.config)
    if isinstance(value, list):
        print("\n".join(str(item) for item in value))
    else:
        print(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read central CometKV path configuration.")
    parser.add_argument("--config", default=None, help=f"Path config JSON. Defaults to ${PATH_CONFIG_ENV} or config/paths.json.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_select = subparsers.add_parser("model-select")
    model_select.add_argument("model")
    model_select.set_defaults(func=_cmd_model_select)

    model_path = subparsers.add_parser("model-path")
    model_path.add_argument("model")
    model_path.set_defaults(func=_cmd_model_path)

    model_names = subparsers.add_parser("model-names")
    model_names.set_defaults(func=_cmd_model_names)

    get = subparsers.add_parser("get")
    get.add_argument("key", help="Dot key, for example datasets.longbench.data_dir")
    get.set_defaults(func=_cmd_get)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
