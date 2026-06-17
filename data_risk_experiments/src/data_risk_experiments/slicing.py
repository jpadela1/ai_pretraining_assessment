"""
Slicing dispatcher.

The loader can be specified two ways:
  - 'module.path:function_name'  (string, imported dynamically — normal case)
  - a callable                    (used by tests so they don't need to be
                                   importable as modules)
"""

from __future__ import annotations
import importlib
from typing import Callable, Union

import pandas as pd


def _resolve_loader(loader_spec: Union[str, Callable]) -> Callable:
    if callable(loader_spec):
        return loader_spec
    module_path, func_name = loader_spec.split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def load_slices(dataset_cfg: dict, cache_dir: str,
                seed: int = 42) -> list[tuple[str, pd.DataFrame]]:
    loader = _resolve_loader(dataset_cfg["loader"])
    return loader(dataset_cfg["slicing"], cache_dir=cache_dir, seed=seed)
