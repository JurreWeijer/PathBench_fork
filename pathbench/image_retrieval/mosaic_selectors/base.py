from typing import Optional, Dict, Any

class MosaicSelector:
    """
    Base class for all mosaic selectors.

    Subclasses should define:
      - name: str
      - HYPERPARAMS: {param_name: {type, default, min, max, choices, help, attr}}
    """

    name: str = ""                 
    HYPERPARAMS: Dict[str, Dict[str, Any]] = {}

    def __init__(self, params, config: dict, **kwargs) -> None:
        self.config = config or {}
        self.params = params or {}
        self.extra  = kwargs or {} 

    @classmethod
    def hyperparam_spec(cls) -> Dict[str, Dict[str, Any]]:
        """Return the hyperparameter schema for this selector."""
        return dict(getattr(cls, "HYPERPARAMS", {}) or {})

    def hyperparam_values(self) -> Dict[str, Any]:
        """
        Return current hyperparameter values for this instance.
        Tries (in order): explicit attribute via spec['attr'] -> attribute by key -> params -> default.
        """
        spec = self.hyperparam_spec()
        out = {}
        for key, meta in spec.items():
            attr = meta.get("attr")
            if attr and hasattr(self, attr):
                out[key] = getattr(self, attr)
            elif hasattr(self, key):
                out[key] = getattr(self, key)
            elif key in (self.params or {}):
                out[key] = self.params[key]
            else:
                out[key] = meta.get("default")
        return out
    
    def _get_hp(self, key: str) -> Any:
        """
        Resolve a hyperparameter:
          1) take from self.params if provided
          2) else fallback to HYPERPARAMS[key]['default'] if present
          3) coerce to declared 'type' and clamp to ['min','max'] or validate 'choices'
        """
        spec = self.hyperparam_spec().get(key, {})
        val = self.params.get(key, spec.get("default"))

        # Type coercion
        typ = spec.get("type")
        if typ is not None and val is not None and not isinstance(val, typ):
            try:
                val = typ(val)
            except Exception:
                raise ValueError(f"Failed to cast hyperparam '{key}' to {typ.__name__}: {val!r}")

        # Choices validation
        choices = spec.get("choices")
        if choices is not None and val is not None and val not in choices:
            raise ValueError(f"Hyperparam '{key}' must be one of {choices}, got {val!r}")

        # Clamp
        if isinstance(val, (int, float)):
            if "min" in spec: val = max(spec["min"], val)
            if "max" in spec: val = min(spec["max"], val)

        return val
    
    def additional_data(self) -> dict:
        """Optional hook. Return a JSON/pickle-serializable dict to store
        under 'additional_data' in the mosaic file. Default: {}."""
        return {}
    
    def run(self, patches: list, **kwargs) -> tuple:
        """Return (selected, group_ids, coords, groups)."""
        raise NotImplementedError
    
