"""Backend registry — runs at import time, registers default backends."""
from typing import Dict
from backends.base import Backend
from backends.qwen import QwenBackend
from backends.spark import SparkBackend

_REGISTRY: Dict[str, Backend] = {}


def register(backend: Backend) -> None:
    _REGISTRY[backend.name] = backend


def get(name: str) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def all_backends() -> Dict[str, Backend]:
    return dict(_REGISTRY)


register(QwenBackend())
register(SparkBackend())
