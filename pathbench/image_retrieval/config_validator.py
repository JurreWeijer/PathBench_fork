import os
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SISConfigValidator:
    # allowed values—everything stored in lower case
    ALLOWED_QC = {"gaussianv2", "otsu-clahe"}
    ALLOWED_DATASET_USE = {"training", "validation", "testing"}
    ALLOWED_VIZ_METHODS = {"umap-mean", "umap-median", "umap-none"}
    ALLOWED_UMAP_KEYS = {"n_neighbors", "min_dist", "metric"}
    ALLOWED_NORMALIZATIONS = {"reinhard", "macenko", "cyclegan"}
    ALLOWED_FEATURE_EXTRACTORS = {
        "resnet50_imagenet", "ctranspath", "transpath_mocov3", "retccl",
        "plip", "histossl", "uni", "uni_h", "conch", "dino", "mocov2",
        "swav", "phikon", "phikon_v2", "gigapath", "barlow_twins", "hibou_b",
        "hibou_l", "pathoduet_ihc", "pathoduet_he", "kaiko_s8", "kaiko_s16",
        "kaiko_b8", "kaiko_b16", "kaiko_l14", "h_optimus_0", "h_optimus_1",
        "virchow", "virchow2", "exaone_path"
    }
    ALLOWED_SEARCH_METHODS = {"yottixel", "sish"}
    ALLOWED_MOSAIC_BASE = {"splice_rgb", "splice_features", "yottixel_rgb", "yottixel_features", "sdm_features"}
    METRIC_PATTERN = re.compile(r"^(hit|mmv|map)_at_\d+$", re.IGNORECASE)

    def __init__(self, config: dict):
        self.cfg = config
        self.errors = []

    def validate(self):
        self._validate_experiment()
        self._validate_datasets()
        self._validate_visualization()
        self._validate_umap_parameters()
        self._validate_benchmark_parameters()
        self._validate_other()
        if self.errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(self.errors))

    def _validate_experiment(self):
        exp = self.cfg.get("experiment", {})
        # required keys
        for key in ("project_name", "annotation_file", "report",
                    "tile_extraction_only", "feature_extraction_only",
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
        for bool_key in ("report", "tile_extraction_only", "feature_extraction_only"):
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
        viz = self.cfg.get("visualization", [])
        if not isinstance(viz, list):
            self.errors.append("`visualization` must be a list")
            return
        for v in viz:
            if v.lower() not in self.ALLOWED_VIZ_METHODS:
                self.errors.append(f"Unknown visualization method: {v}")

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
        for fe in bp.get("feature_extraction", []):
            if fe.lower() not in self.ALLOWED_FEATURE_EXTRACTORS:
                self.errors.append(f"Unsupported feature_extraction: {fe}")

        # search_method
        for sm in bp.get("search_method", []):
            parts = sm.split("-",1)
            if len(parts)!=2 or parts[0].lower() not in self.ALLOWED_SEARCH_METHODS or not parts[1].isdigit():
                self.errors.append(f"Invalid search_method entry: {sm}")

        # mosaic_method
        for mm in bp.get("mosaic_method", []):
            parts = mm.split("-",1)
            base, pct = (parts[0], parts[1]) if len(parts)==2 else (parts[0], None)
            if base.lower() not in self.ALLOWED_MOSAIC_BASE:
                self.errors.append(f"Unsupported mosaic_method base: {mm}")
            if pct and pct.lower()!="none":
                try:
                    float(pct)
                except:
                    self.errors.append(f"Invalid mosaic percentile: {mm}")
        
        if "roi" in bp:
            roi_vals = bp["roi"]
            # must be a list of bools
            if not (isinstance(roi_vals, (list,tuple)) and all(isinstance(v,bool) for v in roi_vals)):
                self.errors.append("`benchmark_parameters.roi` must be a list of booleans")
            # if any experiment wants ROI=True, check that each dataset has a valid roi_path
            if any(roi_vals):
                datasets = self.cfg.get("datasets", [])
                for idx, ds in enumerate(datasets):
                    rp = ds.get("roi_path")
                    if not rp:
                        self.errors.append(f"[datasets][{idx}] missing `roi_path` but ROI=True requested")
                    elif not os.path.isdir(rp):
                        self.errors.append(f"[datasets][{idx}] `roi_path` does not exist: {rp!r}")

    def _validate_other(self):
        wd = self.cfg.get("weights_dir","")
        if not wd or not os.path.isdir(wd):
            self.errors.append(f"`weights_dir` not found or not a dir: {wd}")
        hf = self.cfg.get("hf_key","")
        if not isinstance(hf,str) or not hf.strip():
            self.errors.append("`hf_key` must be a non-empty string")
