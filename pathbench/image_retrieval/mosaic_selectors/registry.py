# registry.py
from __future__ import annotations
from typing import Dict, Type, Optional
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

def get_selector_param_key(name: str) -> Optional[str]:
    """
    Return the selector's `param_key` (e.g., 'percentile_threshold'),
    or None if the selector doesn't use a numeric param.
    """
    cls = _mosaic_selectors.get(name.lower())
    if cls is None:
        return None
    return getattr(cls, "param_key", None)

def build_mosaic_selector(name: str, config: dict):
    """
    Instantiate a registered mosaic selector by **plain name** (no hyphen param).
    """
    cls = _mosaic_selectors.get(name.lower())
    if cls is None:
        available = ", ".join(list_mosaic_selectors())
        raise ValueError(f"Unknown mosaic selector '{name}'. Available: [{available}]")
    return cls(config)
