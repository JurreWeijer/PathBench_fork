# pathbench/image_retrieval/search_methods/base.py
import os

class SearchMethodBase:
    """
    Minimal interface for search implementations.
    Subclasses must set:
      - name: str          (e.g., "yottixel")
      - supports: set[str] (subset of {"patch", "slide"})
    And implement:
      - leave_one_patient_out(self) -> list[dict]
    """

    name: str = ""
    supports: set[str] = frozenset({"patch"})  # default: patch-only

    def __init__(self, config: dict, slide_representation_paths: dict, k: int = 5, **kwargs):
        self.config = config
        self.paths = slide_representation_paths
        self.k = k
        self.mode = self._infer_mode(slide_representation_paths)  # "patch" or "slide"
        if self.mode not in self.supports:
            raise ValueError(
                f"'{self.name}' does not support {self.mode}-level search. "
                f"Supported: {sorted(self.supports)}"
            )

    @staticmethod
    def _infer_mode(paths: dict) -> str:
        exts = {os.path.splitext(p)[1].lower() for p in paths.values()}
        if exts == {".pkl"}:
            return "patch"
        if exts == {".pt"}:
            return "slide"
        raise ValueError(f"Mixed/unknown representation types: {exts}")

    def leave_one_patient_out(self) -> list:
        raise NotImplementedError
