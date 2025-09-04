# registry.py
from __future__ import annotations
from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)

# Internal registry (keys are lowercased aliases)
_patch_selectors: Dict[str, Type] = {}

def _alias_for(cls: Type) -> str:
    return getattr(cls, "name", cls.__name__).lower()

def register_patch_selectors(cls: Type):
    key = _alias_for(cls)
    if key in _patch_selectors:
        raise ValueError(
            f"Patch selector alias '{key}' already registered by "
            f"{_patch_selectors[key].__name__}"
        )
    _patch_selectors[key] = cls
    return cls

def list_patch_selectors() -> list[str]:
    return sorted(_patch_selectors.keys())

def has_patch_selector(name: str) -> bool:
    return name.lower() in _patch_selectors

def get_selector_param_key(name: str) -> Optional[str]:
    """
    Return the selector's `param_key` (e.g., 'percentile_threshold'),
    or None if the selector doesn't use a numeric param.
    """
    cls = _patch_selectors.get(name.lower())
    if cls is None:
        return None
    return getattr(cls, "param_key", None)

def build_patch_selector(name: str, config: dict):
    """
    Instantiate a registered patch selector by **plain name** (no hyphen param).
    """
    cls = _patch_selectors.get(name.lower())
    if cls is None:
        available = ", ".join(list_patch_selectors())
        raise ValueError(f"Unknown patch selector '{name}'. Available: [{available}]")
    return cls(config)
