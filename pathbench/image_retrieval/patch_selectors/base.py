from typing import Optional

class PatchSelector:
    """
    Base class for all patch selectors.

    Subclasses must define:
      - name: primary alias (e.g. "SPLICE_rgb")
      - param_key: name of numeric hyphen param or None if not used
    """

    name: str = ""                 # e.g. "SPLICE_rgb"
    param_key: Optional[str] = None  # e.g. "percentile_threshold" or None

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, patches: list, **kwargs) -> tuple:
        """Return (selected, group_ids, coords, groups)."""
        raise NotImplementedError