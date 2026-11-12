import builtins
import importlib
import sys


def test_model_imports_do_not_require_weighted_attention_backend(monkeypatch):
    for module_name in list(sys.modules):
        if (
            module_name == "model_hub"
            or module_name.startswith("model_hub.")
            or module_name == "cache_hub"
            or module_name.startswith("cache_hub.")
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "weighted_flash_decoding" or name.startswith("weighted_flash_decoding."):
            raise AssertionError("model imports should not load weighted attention backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    model_hub = importlib.import_module("model_hub")

    assert model_hub.LlamaModel.__name__ == "LlamaModel"
    assert model_hub.QwenModel.__name__ == "QwenModel"
