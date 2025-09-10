import os
import re
import logging
import importlib
import pkgutil

from difflib import get_close_matches

from .mosaic_selectors.registry import list_mosaic_selectors, get_mosaic_selector_hyperparams, has_mosaic_selector
from .search_methods.registry import list_search_methods, get_search_method_supports, get_search_method_hyperparams
from slideflow.model.extractors._registry import list_extractors as sf_list_extractors, is_extractor as sf_is_extractor

import importlib, logging
try:
    importlib.import_module("pathbench.models.feature_extractors")
    logging.debug("Imported pathbench.models.feature_extractors")
except Exception as e:
    logging.warning(f"Could not import PathBench extractors: {e}")

logger = logging.getLogger(__name__)

class SISConfigValidator:
    # allowed values—everything stored in lower case
    ALLOWED_QC = {"gaussianv2", "otsu-clahe"}
    ALLOWED_DATASET_USE = {"training", "validation", "testing"}
    ALLOWED_VIZ_METHODS = {"umap", "patch_selection", "retrieval_report"}
    ALLOWED_UMAP_KEYS = {"n_neighbors", "min_dist", "metric"}
    ALLOWED_NORMALIZATIONS = {"reinhard", "macenko", "cyclegan"}

    METRIC_PATTERN = re.compile(r"^(hit|mmv|map)_at_\d+$", re.IGNORECASE)

    def __init__(self, config: dict):
        self.cfg = config
        self.errors = []
        self._combination_error = ""

    def validate(self):
        self._validate_experiment()
        self._validate_datasets()
        self._validate_visualization()
        self._validate_umap_parameters()
        self._validate_benchmark_parameters()
        self._validate_other()
        if self.errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(self.errors))
    
    def _parse_name_params_list(self, items, available_names, section_name):
        """
        Normalize a list where each item is either:
        - "name"                         -> (name, {})
        - "name-<k>"  (legacy search)    -> (name, {"k": int(k)})
        - {name: {param: value, ...}}    -> (name, params)
        - {name: null}                   -> (name, {})
        Names are lower-cased. Validates that name is in available_names.
        """
        out = []
        if not isinstance(items, list):
            self.errors.append(f"`benchmark_parameters.{section_name}` must be a list")
            return out

        for i, item in enumerate(items):
            name, params = None, {}
            if isinstance(item, str):
                s = item.strip().lower()
                if section_name == "search_method" and "-" in s and s.rsplit("-", 1)[1].isdigit():
                    base, kstr = s.rsplit("-", 1)
                    name, params = base, {"k": int(kstr)}
                else:
                    name = s
            elif isinstance(item, dict):
                if len(item) != 1:
                    self.errors.append(
                        f"[benchmark_parameters.{section_name}][{i}] must be a single-key mapping; got {item!r}"
                    ); continue
                (raw_name, raw_params), = item.items()
                name = str(raw_name).lower()
                if raw_params is None:
                    params = {}
                elif isinstance(raw_params, dict):
                    params = raw_params
                else:
                    self.errors.append(
                        f"[benchmark_parameters.{section_name}][{i}] params must be a mapping or null; got {type(raw_params).__name__}"
                    ); continue
            else:
                self.errors.append(
                    f"[benchmark_parameters.{section_name}][{i}] must be a string or single-key mapping; got {type(item).__name__}"
                ); continue

            if name not in available_names:
                self.errors.append(
                    f"Unknown {section_name.rstrip('s')} '{name}'. Available: {sorted(available_names)}"
                ); continue

            out.append((name, params))
        return out

    def _validate_experiment(self):
        exp = self.cfg.get("experiment", {})
        # required keys
        for key in ("project_name", "annotation_file", "report",
                    "skip_extracted", "skip_feature_extraction", "save_tiles", "mixed_precision",
                    "num_workers", "qc", "qc_filters", "evaluation"):
            if key not in exp:
                self.errors.append(f"[experiment] missing required key: '{key}'")

        # project_name
        if not isinstance(exp.get("project_name",""), str) or not exp["project_name"].strip():
            self.errors.append("`experiment.project_name` must be a non-empty string")

        # annotation_file
        ann = exp.get("annotation_file")
        if not exp.get("feature_extraction_only", False):
            if not ann or not os.path.isfile(ann):
                self.errors.append(f"`experiment.annotation_file` not found: {ann}")

        # booleans
        for bool_key in ("report", "skip_extracted", "skip_feature_extraction", "save_tiles", "mixed_precision", "resume"):
            if not isinstance(exp.get(bool_key), bool):
                self.errors.append(f"`experiment.{bool_key}` must be boolean")

        # num_workers
        nw = exp.get("num_workers")
        if not isinstance(nw, int) or nw < 1:
            self.errors.append("`experiment.num_workers` must be a positive integer")

        # QC methods (case‐insensitive)
        qc_methods = exp.get("qc")
        if not isinstance(qc_methods, list):
            qc_methods = []

        if len(qc_methods) > 0:
            for qc in qc_methods:
                if qc.lower() not in self.ALLOWED_QC:
                    self.errors.append(f"Unknown QC method: {qc}")

        # QC filters
        qf = exp.get("qc_filters", {})
        for f in ("grayspace_threshold","grayspace_fraction",
                  "whitespace_threshold","whitespace_fraction"):
            if f not in qf:
                self.errors.append(f"[experiment.qc_filters] missing '{f}'")
        # numeric ranges
        try:
            gt, gf = float(qf["grayspace_threshold"]), float(qf["grayspace_fraction"])
            wt, wf = float(qf["whitespace_threshold"]), float(qf["whitespace_fraction"])
            if not (0 <= gt <= 1 and 0 <= gf <= 1 and 0 <= wt <= 255 and 0 <= wf <= 1):
                raise ValueError
        except Exception:
            self.errors.append("One or more QC filter values out of valid range")

        # evaluation metrics, allow any case
        ev = exp.get("evaluation", [])
        if not isinstance(ev, list) or not ev:
            self.errors.append("`experiment.evaluation` must be a non-empty list")
        else:
            for m in ev:
                if not isinstance(m, str) or not self.METRIC_PATTERN.match(m):
                    self.errors.append(f"Invalid metric name: {m}")

    def _validate_datasets(self):
        ds_list = self.cfg.get("datasets")
        if not isinstance(ds_list, list) or not ds_list:
            self.errors.append("`datasets` must be a non-empty list")
            return

        for idx, ds in enumerate(ds_list):
            # Required keys remain errors
            for key in ("name", "slide_path", "tfrecord_path", "tile_path", "used_for"):
                if key not in ds:
                    self.errors.append(f"[datasets][{idx}] missing required key: '{key}'")

            # Existence checks are now warnings
            for path_key in ("slide_path", "tfrecord_path", "tile_path"):
                p = ds.get(path_key)
                if not p or not os.path.isdir(p):
                    logging.warning(f"[datasets][{idx}] directory not found (skipping strict validation): "
                                    f"{path_key}={p!r}")

            # used_for still an error if invalid
            uf = ds.get("used_for")
            if uf not in self.ALLOWED_DATASET_USE:
                self.errors.append(
                    f"[datasets][{idx}].used_for must be one of {self.ALLOWED_DATASET_USE}, got {uf!r}"
                )

    def _validate_visualization(self):
        viz = self.cfg["experiment"].get("visualization", {})
        if not isinstance(viz, dict):
            self.errors.append("`experiment.visualization` must be a mapping (dict).")
            return

        allowed_sections = {"patch_selection", "umap", "retrieval_report"}
        for section in viz.keys():
            if section not in allowed_sections:
                self.errors.append(
                    f"Unknown visualization section `{section}`. "
                    f"Allowed: {sorted(allowed_sections)}"
                )

        # ---- patch_selection ----
        if "patch_selection" in viz:
            ps = viz["patch_selection"]
            if not isinstance(ps, dict):
                self.errors.append("`experiment.visualization.patch_selection` must be a dict.")
            else:
                mode = ps.get("mode")
                valid_modes = {"extensive", "simple"}
                if mode is None:
                    self.errors.append(
                        "`experiment.visualization.patch_selection.mode` is required "
                        "and must be 'extensive' or 'simple'."
                    )
                elif str(mode).lower() not in valid_modes:
                    self.errors.append(
                        f"Invalid `patch_selection.mode`: {mode!r}. "
                        f"Allowed: {sorted(valid_modes)}"
                    )
                if "max_per_file" in ps:
                    mpf = ps["max_per_file"]
                    if not isinstance(mpf, int) or mpf <= 0:
                        self.errors.append(
                            "`experiment.visualization.patch_selection.max_per_file` "
                            "must be a positive integer."
                        )

        # ---- umap ----
        if "umap" in viz:
            um = viz["umap"]
            if not isinstance(um, dict):
                self.errors.append("`experiment.visualization.umap` must be a dict.")
            else:
                agg = um.get("agg_methods")
                valid_aggs = {"mean", "median", "none"}
                if agg is None:
                    self.errors.append(
                        "`experiment.visualization.umap.agg_methods` is required "
                        "and must be a list of 'mean', 'median', or 'none'."
                    )
                elif not isinstance(agg, (list, tuple, set)):
                    self.errors.append(
                        "`experiment.visualization.umap.agg_methods` must be a list/tuple/set of strings."
                    )
                else:
                    for i, a in enumerate(agg):
                        if not isinstance(a, str) or a.lower() not in valid_aggs:
                            self.errors.append(
                                f"Invalid `umap.agg_methods[{i}]` = {a!r}. "
                                f"Allowed: {sorted(valid_aggs)}"
                            )
                # parameters are optional; allow dict or list (your YAML uses a list of single-key dicts)
                params = um.get("parameters")
                if params is not None and not isinstance(params, (list, dict)):
                    self.errors.append(
                        "`experiment.visualization.umap.parameters` must be a dict or list."
                    )

        # ---- retrieval_report (optional) ----
        if "retrieval_report" in viz:
            rr = viz["retrieval_report"]
            if not isinstance(rr, dict):
                self.errors.append("`experiment.visualization.retrieval_report` must be a dict.")
            else:
                if "include_metadata" in rr and not isinstance(rr["include_metadata"], bool):
                    self.errors.append(
                        "`experiment.visualization.retrieval_report.include_metadata` must be a boolean."
                    )
                if "max_per_file" in rr:
                    mpf = rr["max_per_file"]
                    if not isinstance(mpf, int) or mpf <= 0:
                        self.errors.append(
                            "`experiment.visualization.retrieval_report.max_per_file` must be a positive integer."
                        )

    def _validate_umap_parameters(self):
        params = self.cfg.get("umap_parameters", [])
        if not isinstance(params, list):
            self.errors.append("`umap_parameters` must be a list of single-key dicts")
            return
        seen = set()
        for d in params:
            if not isinstance(d, dict) or len(d)!=1:
                self.errors.append(f"Each umap_parameters entry must be a single-key dict: {d}")
                continue
            k, v = next(iter(d.items()))
            if k.lower() not in self.ALLOWED_UMAP_KEYS:
                self.errors.append(f"Unknown umap parameter: {k}")
            if k in seen:
                self.errors.append(f"Duplicate umap parameter: {k}")
            seen.add(k)
            if k in ("n_neighbors","min_dist") and not isinstance(v, (int,float)):
                self.errors.append(f"UMAP parameter '{k}' must be numeric")
            if k=="metric" and not isinstance(v, str):
                self.errors.append("UMAP parameter 'metric' must be a string")

    def _validate_benchmark_parameters(self):
        bp = self.cfg.get("benchmark_parameters", {})
        # tile_px
        tpx = bp.get("tile_px")
        if not (isinstance(tpx, (list,tuple)) and all(isinstance(x,int) and x>0 for x in tpx)):
            self.errors.append("benchmark_parameters.tile_px must be a list of positive ints")

        # tile_um
        tum = bp.get("tile_um")
        pattern = re.compile(r"\d+(\.\d+)?x$", re.IGNORECASE)

        if not (
            isinstance(tum, (list, tuple))
            and all(
                (isinstance(u, int) and u > 0)
                or (isinstance(u, str) and pattern.fullmatch(u.strip()))
                for u in tum
            )
        ):
            self.errors.append(
                "`benchmark_parameters.tile_um` must be a list of either positive ints "
                "(microns) or strings like '20x', '40x', etc."
            )

        # normalization (case‐insensitive)
        for norm in bp.get("normalization", []):
            if norm.lower() not in self.ALLOWED_NORMALIZATIONS:
                self.errors.append(f"Unknown normalization: {norm}")

        # feature_extraction (case‐insensitive)
        available_extractors = sf_list_extractors()
        for feature_extractor in bp.get("feature_extraction", []):
            if not sf_is_extractor(feature_extractor):
                # nice error with suggestions
                sugg = get_close_matches(feature_extractor, available_extractors, n=3)
                hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
                self.errors.append(
                    f"Unsupported feature_extraction: {feature_extractor}.{hint} "
                    f"Available: {sorted(available_extractors)}"
                )

        # ---- search_method: only check that provided param names exist in schema ----
        available_methods = set(list_search_methods())
        parsed_search = self._parse_name_params_list(
            bp.get("search_method", []), available_methods, "search_method"
        )
        for i, (method, params) in enumerate(parsed_search):
            # fetch schema declared by the method
            schema = get_search_method_hyperparams(method) or {}
            allowed_keys = set(schema.keys())
            # allow empty params; only flag unknown keys
            unknown = set(params.keys()) - allowed_keys
            if unknown:
                self.errors.append(
                    f"[benchmark_parameters.search_method][{i}] '{method}' got unknown param(s): "
                    f"{sorted(unknown)}. Allowed: {sorted(allowed_keys)}"
                )

        # ---- mosaic_selector: only check that provided param names exist in schema ----
        available_selectors = set(list_mosaic_selectors())
        parsed_selectors = self._parse_name_params_list(
            bp.get("mosaic_selector", []), available_selectors, "mosaic_selector"
        )
        for i, (selector, params) in enumerate(parsed_selectors):
            schema = get_mosaic_selector_hyperparams(selector) or {}
            allowed_keys = set(schema.keys())
            unknown = set(params.keys()) - allowed_keys
            if unknown:
                self.errors.append(
                    f"[benchmark_parameters.mosaic_selector][{i}] '{selector}' got unknown param(s): "
                    f"{sorted(unknown)}. Allowed: {sorted(allowed_keys)}"
                )
        
    def _validate_other(self):
        wd = self.cfg.get("weights_dir","")
        if not wd or not os.path.isdir(wd):
            self.errors.append(f"`weights_dir` not found or not a dir: {wd}")
        hf = self.cfg.get("hf_key","")
        if not isinstance(hf,str) or not hf.strip():
            self.errors.append("`hf_key` must be a non-empty string")

    def is_slide_extractor(self, extractor_name: str) -> bool:
        """
        Returns True if the given extractor is a slide‐level feature extractor.
        """
        return extractor_name.lower().endswith("_slide")
    
    def is_patch_extractor(self, extractor_name: str) -> bool:
        """
        Returns True if the given extractor is a patch‐level feature extractor.
        """
        return not self.is_slide_extractor(extractor_name)

    def is_patch_search(self, method: str) -> bool:
        supports = get_search_method_supports(method.lower())
        return bool(supports and "patch" in supports)

    def is_slide_search(self, method: str) -> bool:
        supports = get_search_method_supports(method.lower())
        return bool(supports and "slide" in supports)
    
    def is_valid_combination(self, combo: dict) -> bool:
        """
        Enforces that:
          - patch‐models only go with patch‐searches
          - slide‐models only go with slide‐searches
        """
        feat = combo.get("feature_extraction", "").lower()
        search = combo.get("search_method", "").lower()
        
        # patch‐level extractor must use a patch‐level search
        if self.is_patch_extractor(feat) and not self.is_patch_search(search):
            self._combination_error = (
                f"Extractor '{feat}' is patch‐level, but search method '{search}' "
                "is not patch‐level."
            )
            return False

        # slide‐level extractor must use a slide‐level search
        if self.is_slide_extractor(feat) and not self.is_slide_search(search):
            self._combination_error = (
                f"Extractor '{feat}' is slide‐level, but search method '{search}' "
                "is not slide‐level."
            )
            return False

        # if we get here, it’s valid
        self._combination_error = ""
        return True

    @property
    def combination_error(self) -> str:
        """Last reason why validation failed (or empty if none)."""
        return self._combination_error