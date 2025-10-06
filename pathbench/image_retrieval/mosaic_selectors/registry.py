# registry.py
from __future__ import annotations
from typing import Dict, Type, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Internal registry (keys are lowercased aliases)
_mosaic_selectors: Dict[str, Type] = {}

def _alias_for(cls: Type) -> str:
    return getattr(cls, "name", cls.__name__).lower()

def register_mosaic_selectors(cls: Type):
    key = _alias_for(cls)
    if key in _mosaic_selectors:
        raise ValueError(
            f"Mosaic selector alias '{key}' already registered by "
            f"{_mosaic_selectors[key].__name__}"
        )
    _mosaic_selectors[key] = cls
    return cls

def list_mosaic_selectors() -> list[str]:
    return sorted(_mosaic_selectors.keys())

def has_mosaic_selector(name: str) -> bool:
    return name.lower() in _mosaic_selectors

def get_mosaic_selector_class(name: str) -> Type:
    cls = _mosaic_selectors.get(name.lower())
    if cls is None:
        available = ", ".join(list_mosaic_selectors())
        raise ValueError(f"Unknown mosaic selector '{name}'. Available: [{available}]")
    return cls

def build_mosaic_selector(name: str, params: dict, config: dict, **kwargs):
    """Instantiate a registered mosaic selector by **plain name**."""
    cls = get_mosaic_selector_class(name)
    return cls(params, config, **kwargs)

def get_mosaic_selector_hyperparams(name: str) -> dict:
    cls = get_mosaic_selector_class(name)
    return getattr(cls, "HYPERPARAMS", {}) or {}

def get_mosaic_selector_defaults(name: str) -> Dict[str, Any]:
    """Return {param: default} from the schema."""
    spec = get_mosaic_selector_hyperparams(name)
    return {k: v.get("default") for k, v in spec.items()}

def get_mosaic_selector_values(name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return effective values (defaults merged with provided params) WITHOUT instantiation.
    Used for building ID strings safely.
    """
    spec = get_mosaic_selector_hyperparams(name)
    params = params or {}
    out: Dict[str, Any] = {}
    for k, meta in spec.items():
        out[k] = params.get(k, meta.get("default"))
    return out

def describe_all_mosaic_selectors() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Convenience: return {selector_name: schema} for all registered selectors."""
    return {name: getattr(cls, "hyperparam_spec", lambda: getattr(cls, "HYPERPARAMS", {}))()
            for name, cls in _mosaic_selectors.items()}
