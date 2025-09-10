# search_methods/base.py
import os
from typing import Any, Dict

class SearchMethodBase:
    """
    Subclasses should define:
      - name: str
      - supports: set[str]  (subset of {"patch","slide"})
      - HYPERPARAMS: {param: {type, default, min, max, choices, help, attr?}} (optional)
    """
    name: str = ""
    supports: set[str] = frozenset({"patch"})
    HYPERPARAMS: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        config: dict,
        slide_representation_paths: dict,
        params: Dict[str, Any],  
        **kwargs
    ):
        if params is None or not isinstance(params, dict):
            raise TypeError("SearchMethodBase requires a 'params' dict (got None or non-dict).")

        self.config = config or {}
        self.paths  = slide_representation_paths or {}
        self.params = params
        self.mode = self._infer_mode(self.paths)

        for key, meta in self.hyperparam_spec().items():
            attr = meta.get("attr") or key
            setattr(self, attr, self._get_hp(key))

        if self.mode not in self.supports:
            raise ValueError(
                f"'{self.name}' does not support {self.mode}-level search. "
                f"Supported: {sorted(self.supports)}"
            )

    # ---------- hyperparameter helpers ----------
    @classmethod
    def hyperparam_spec(cls) -> Dict[str, Dict[str, Any]]:
        return dict(getattr(cls, "HYPERPARAMS", {}) or {})

    def _get_hp(self, key: str) -> Any:
        spec = self.hyperparam_spec().get(key, {})
        val = self.params.get(key, spec.get("default"))

        typ = spec.get("type")
        if typ is not None and val is not None and not isinstance(val, typ):
            try:
                val = typ(val)
            except Exception:
                raise ValueError(f"Failed to cast hyperparam '{key}' to {typ.__name__}: {val!r}")

        choices = spec.get("choices")
        if choices is not None and val is not None and val not in choices:
            raise ValueError(f"Hyperparam '{key}' must be one of {choices}, got {val!r}")

        if isinstance(val, (int, float)):
            if "min" in spec: val = max(spec["min"], val)
            if "max" in spec: val = min(spec["max"], val)
        return val

    def hyperparam_values(self) -> Dict[str, Any]:
        spec = self.hyperparam_spec()
        out: Dict[str, Any] = {}
        for key, meta in spec.items():
            attr = meta.get("attr")
            if attr and hasattr(self, attr):
                out[key] = getattr(self, attr)
            elif hasattr(self, key):
                out[key] = getattr(self, key)
            elif key in self.params:
                out[key] = self.params[key]
            else:
                out[key] = meta.get("default")
        return out

    # ---------- core API ----------
    @staticmethod
    def _infer_mode(paths: dict) -> str:
        exts = {os.path.splitext(p)[1].lower() for p in paths.values()}
        if exts == {".pkl"}: return "patch"
        if exts == {".pt"}:  return "slide"
        raise ValueError(f"Mixed/unknown representation types: {exts}")

    def leave_one_patient_out(self) -> list:
        raise NotImplementedError
