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

def build_mosaic_selector(name: str, params: dict, config: dict):
    """
    Instantiate a registered mosaic selector by **plain name** (no hyphen param).
    """
    cls = _mosaic_selectors.get(name.lower())
    if cls is None:
        available = ", ".join(list_mosaic_selectors())
        raise ValueError(f"Unknown mosaic selector '{name}'. Available: [{available}]")
    return cls(params, config)

# registry.py
from typing import Dict, Any, Optional

def get_mosaic_selector_hyperparams(name: str) -> Dict[str, Dict[str, Any]]:
    """
    Return the class-level hyperparameter schema for a selector.
    """
    cls = _mosaic_selectors.get(name.lower())
    if cls is None:
        available = ", ".join(list_mosaic_selectors())
        raise ValueError(f"Unknown mosaic selector '{name}'. Available: [{available}]")
    return cls.hyperparam_spec()

def get_mosaic_selector_defaults(name: str) -> Dict[str, Any]:
    """
    Return {param: default} from the schema.
    """
    spec = get_mosaic_selector_hyperparams(name)
    return {k: v.get("default") for k, v in spec.items()}

def get_mosaic_selector_values(name: str, params: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Instantiate (lightweight) and return the *effective* values (params merged with defaults).
    """
    cls = _mosaic_selectors.get(name.lower())
    if cls is None:
        available = ", ".join(list_mosaic_selectors())
        raise ValueError(f"Unknown mosaic selector '{name}'. Available: [{available}]")
    inst = cls(params or {}, config or {})
    return inst.hyperparam_values()

def describe_all_mosaic_selectors() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Convenience: return {selector_name: schema} for all registered selectors.
    """
    return {name: cls.hyperparam_spec() for name, cls in _mosaic_selectors.items()}
