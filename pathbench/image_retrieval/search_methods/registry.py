# registry.py
from __future__ import annotations
from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)

# Internal registry (keys are lowercased aliases)
_search_methods: Dict[str, Type] = {}

def _alias_for(cls: Type) -> str:
    return getattr(cls, "name", cls.__name__).lower()

def register_search_methods(cls: Type):
    key = _alias_for(cls)
    if key in _search_methods:
        raise ValueError(
            f"Search method alias '{key}' already registered by "
            f"{_search_methods[key].__name__}"
        )
    _search_methods[key] = cls
    return cls

def list_search_methods() -> list[str]:
    return sorted(_search_methods.keys())

def has_search_method(name: str) -> bool:
    return name.lower() in _search_methods

def build_search_method(name: str, **kwargs):
    _autoload_all_search_methods()  # ensure everything is imported before lookup
    cls = _search_methods.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown search method '{name}'. Available: {list_search_methods()}")
    return cls(**kwargs)

# ---- recursive autoloader (works with subpackages, no __init__.py required) ----
def _autoload_all_search_methods():
    """Recursively import all *_search modules so decorators execute."""
    import pkgutil, importlib, fnmatch
    # This module is ...search_methods.registry → we want the package ...search_methods
    pkg_name = __name__.rsplit(".", 1)[0]
    pkg = importlib.import_module(pkg_name)

    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        # Only import files that look like our implementations (e.g., yottixel_search.py)
        if not fnmatch.fnmatch(modinfo.name.split(".")[-1], "*_search"):
            # still import subpackages so walk continues into them
            # (safe: importing a package does not execute heavy code)
            if modinfo.ispkg:
                try:
                    importlib.import_module(modinfo.name)
                except Exception as e:
                    logger.debug(f"Skipping import of package {modinfo.name}: {e}")
            continue
        try:
            importlib.import_module(modinfo.name)
        except Exception as e:
            logger.warning(f"Failed to import {modinfo.name}: {e}")

# Optionally trigger once on import (harmless if left enabled)
_autoload_all_search_methods()
