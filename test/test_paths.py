from config.paths import resolve_model_config


def test_resolve_model_config_expands_tokenizer_path_for_alias():
    model_config = resolve_model_config("Llama-3.1-8B-Instruct")

    assert model_config["name"] == "llama-3.1-8b"
    assert model_config["tokenizer_path"] == model_config["path"]
