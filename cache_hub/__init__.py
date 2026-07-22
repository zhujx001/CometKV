_CACHE_CLASS_MODULES = {
    "flash_attn_cache": ".flash_attn_cache",
    "cometkv_cache": ".cometkv_cache",
    "exact_topk_cache": ".exact_topk_cache",
}

__all__ = tuple(_CACHE_CLASS_MODULES)


def __getattr__(name):
    if name not in _CACHE_CLASS_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_CACHE_CLASS_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
