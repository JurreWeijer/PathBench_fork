# registry.py
from __future__ import annotations

import logging
import pkgutil
import importlib
import fnmatch
from typing import Dict, Type, Optional, Set, Any

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
    _autoload_all_search_methods()
    return sorted(_search_methods.keys())


def has_search_method(name: str) -> bool:
    _autoload_all_search_methods()
    return name.lower() in _search_methods


def get_search_method_supports(name: str) -> Optional[Set[str]]:
    """
    Return declared supported modes for a search method (e.g., {'patch'}, {'slide'}, or both).
    None if unknown.
    """
    _autoload_all_search_methods()
    cls = _search_methods.get(name.lower())
    if cls is None:
        return None
    supports = getattr(cls, "supports", set())
    return {str(s).lower() for s in supports}


# ---------------- Hyperparameter helpers (schema-based) ----------------

def get_search_method_hyperparams(name: str) -> Dict[str, Dict[str, Any]]:
    """
    Return the class-level hyperparameter schema (HYPERPARAMS) for a method.
    """
    _autoload_all_search_methods()
    cls = _search_methods.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown search method '{name}'. Available: {list_search_methods()}")
    # subclasses follow the same API as MosaicSelector: hyperparam_spec()
    if hasattr(cls, "hyperparam_spec"):
        return cls.hyperparam_spec()  # type: ignore[attr-defined]
    # fallback to raw HYPERPARAMS if no helper present
    return dict(getattr(cls, "HYPERPARAMS", {}) or {})


def get_search_method_defaults(name: str) -> Dict[str, Any]:
    """
    Return {param: default} from the schema.
    """
    spec = get_search_method_hyperparams(name)
    return {k: v.get("default") for k, v in spec.items()}


def get_search_method_values(name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return effective values (defaults merged with provided params) WITHOUT instantiating.
    Useful when you don't have slide representations to infer mode yet.
    """
    spec = get_search_method_hyperparams(name)
    params = params or {}
    out: Dict[str, Any] = {}
    for k, meta in spec.items():
        out[k] = params.get(k, meta.get("default"))
    return out


def describe_all_search_methods() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Convenience: {method_name: schema} for all registered methods.
    """
    _autoload_all_search_methods()
    return {name: get_search_method_hyperparams(name) for name in _search_methods}


# ---------------- Constructor ----------------

def build_search_method(
    name: str,
    config: dict,
    slide_representation_paths: dict,
    params: Dict[str, Any],
    **kwargs
):
    """
    Instantiate a registered search method.

    REQUIRED:
      - params: dict   (k and other hyperparameters live here)
    """
    _autoload_all_search_methods()
    cls = _search_methods.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown search method '{name}'. Available: {list_search_methods()}")

    if params is None or not isinstance(params, dict):
        raise TypeError("build_search_method requires a 'params' dict (got None or non-dict).")

    # The new base expects `params=`; pass through any extra kwargs if subclasses use them.
    return cls(
        config=config,
        slide_representation_paths=slide_representation_paths,
        params=params,
        **kwargs
    )


# ---------------- Autoloader ----------------

def _autoload_all_search_methods():
    """Recursively import all modules whose names end with *_search so decorators run."""
    # This module is ...search_methods.registry → we want the package ...search_methods
    pkg_name = __name__.rsplit(".", 1)[0]
    pkg = importlib.import_module(pkg_name)

    for modinfo in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        # Import files that look like implementations, e.g., yottixel_search.py
        leaf = modinfo.name.split(".")[-1]
        if not fnmatch.fnmatch(leaf, "*_search"):
            # Still import subpackages so recursion continues
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


# Optionally trigger once on import (safe)
_autoload_all_search_methods()
