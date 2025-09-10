import slideflow as sf
import numpy as np
import joblib
from typing import List, Dict, Optional, Tuple, Any
from shapely.geometry import Polygon, box
import os
import pandas as pd
import json
import torch
import logging
import hashlib
import re

# Import the registry resolvers (no instantiation happens in these)
from pathbench.image_retrieval.search_methods.registry import get_search_method_values
from pathbench.image_retrieval.mosaic_selectors.registry import get_mosaic_selector_values

from itertools import product

logger = logging.getLogger(__name__)

def load_polygons_for_slide(
    roi_folder: str,
    slide_id: str
) -> Tuple[List[Polygon], Optional[str]]:
    """
    Look in `roi_folder` for either slide_id.geojson or slide_id.csv,
    and return a tuple (polygons, used_path).

    It will only load **one** of them (geojson preferred).  
    If neither exists, returns ([], None).

    Args:
        roi_folder: directory containing your ROI files
        slide_id:    base name of the slide (no extension)

    Returns:
        polygons:    list of Shapely Polygon objects
        used_path:   the path that was loaded ('.geojson' or '.csv'), or None
    """
    # ---- Try geojson first ----
    geojson_path = os.path.join(roi_folder, f"{slide_id}.geojson")
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            gj = json.load(f)
        polys = [
            Polygon(feat["geometry"]["coordinates"][0])
            for feat in gj.get("features", [])
        ]
        return polys, geojson_path

    # ---- Fallback to CSV ----
    csv_path = os.path.join(roi_folder, f"{slide_id}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        polys = []
        for _, group in df.groupby("ROI_Name"):
            coords = list(zip(group["X_base"], group["Y_base"]))
            polys.append(Polygon(coords))
        return polys, csv_path

    # ---- Neither found ----
    return [], None

def load_patch_dicts_from_tfr_roi(
    config: dict,
    tfr_path: str,
    features_index_path: str,      
    features_path: str, 
    roi: bool,       
    patch_size: int
) -> List[Dict]:
    """
    Load patch dictionaries from a TFRecord and corresponding feature index.

    Matches each image patch in the TFRecord to its feature vector using
    spatial location. Computes RGB histograms for color-based filtering or selection.

    Args:
        config (dict): full experiment config (so we can look up `roi_path`).
        tfr_path (str): path to a single TFRecord file.
        features_index_npz_path (str): `.npz` file containing tile-center coords.
        features_path (str): `.pt` or numpy file containing your NxF feature array.
        patch_size (int): tile size (px) for computing grid indices and ROIs.

    Returns:
        dict: {
            "properties": {
                "tfr_path": …,
                "features_index_path": …,
                "features_path": …,
                "roi_path": … or None
            },
            "patches": [ { … }, … ]
        }
    """
    # ---- Load data ----
    tfr_record_iter = sf.TFRecord(tfr_path)
    features_index_npz = np.load(features_index_path)
    features_tensor = torch.load(features_path)

    # always define these
    slide_polys = []
    roi_path = None

    if roi:
        # ---- find matching dataset so we know its roi_folder ----
        roi_folder: Optional[str] = None
        slide_id = os.path.splitext(os.path.basename(tfr_path))[0]
        for dataset in config.get("datasets", []):
            root = dataset.get("tfrecord_path")
            if root and tfr_path.startswith(root):
                roi_folder = dataset.get("roi_path")
                break

        # ---- read roi parameters ----
        roi_method = config.get("roi_parameters", {}).get("roi_method", "inside")
        roi_threshold = float(config.get("roi_parameters", {}).get("roi_filter_method", 0.5))

        # ---- load that slide’s polygons ----
        slide_polys: List[Polygon] = []
        roi_path: Optional[str] = None
        if roi_folder:
            slide_polys, roi_path = load_polygons_for_slide(roi_folder, slide_id)

    feature_locations = features_index_npz["arr_0"]
    features_array    = (features_tensor.numpy() if hasattr(features_tensor, "numpy") else features_tensor)

    output_patches: List[Dict] = []
    for tile_idx, rec in enumerate(tfr_record_iter):
        x_loc = float(rec["loc_x"]); y_loc = float(rec["loc_y"])

        # --- ROI filter if any polygons present ---
        if slide_polys:
            tile_box = box(x_loc, y_loc, x_loc + patch_size, y_loc + patch_size)
            area = tile_box.area
            inter = sum(tile_box.intersection(p).area for p in slide_polys)
            frac  = (inter/area) if area > 0 else 0.0

            if roi_method == "inside" and frac < roi_threshold: 
                continue
            if roi_method == "outside" and (1-frac) < roi_threshold: 
                continue

        # --- match to features ---
        coords = np.array([int(x_loc), int(y_loc)])
        matches = np.where((feature_locations == coords).all(axis=1))[0]
        if matches.size == 0:
            continue
        feat_idx = int(matches[0])

        # --- build patches dict ---
        img = sf.io.decode_image(bytes(rec["image_raw"]))
        rgb = np.array(img).reshape(-1,3).mean(axis=0)/255.0

        output_patches.append({
            "tfr_index": tile_idx,
            "loc": [x_loc, y_loc],
            "wsi_loc": [int(x_loc//patch_size), int(y_loc//patch_size)],
            "feature": features_array[feat_idx],
            "rgb_histogram": rgb
        })
    
    return {
        "properties": {
            "tfr_path":                tfr_path,
            "features_index_path":     features_index_path,
            "features_path":           features_path,
            "roi_path":                roi_path
        },
        "patches": output_patches
    }

def load_patch_dicts_from_tfr(
    tfr_path: str,
    features_index_path: str,
    features_path: str,
    patch_size: int
) -> Dict[str, object]:
    """
    Load every TFRecord tile, match by (x,y) to the features array via a prebuilt dictionary,
    and return a list of patch‐dicts each containing { 'tfr_index', 'loc', 'wsi_loc', 'feature', 'rgb_histogram' }.

    This does an O(N) pass to build coord→feat_idx, then O(M) lookups for M TFRecord tiles.
    """

    # ---- 1) Load feature locations and the feature tensor ----
    features_index_npz = np.load(features_index_path)
    feature_locations  = features_index_npz["arr_0"]   # shape = (N, 2), each row = (x_i, y_i)

    features_tensor = torch.load(features_path)        # shape = (N, D)
    if hasattr(features_tensor, "numpy"):
        features_array = features_tensor.numpy()       # convert to (N, D) numpy
    else:
        features_array = features_tensor               # already numpy

    N_feat = feature_locations.shape[0]
    if features_array.shape[0] != N_feat:
        raise RuntimeError(
            f"Mismatch: feature_locations has {N_feat} rows, but features_array has {features_array.shape[0]} rows."
        )

    # ---- 2) Build coord→feat_idx dictionary in one pass (O(N_feat)) ----
    coord_to_featidx: Dict[tuple, int] = {}
    for idx in range(N_feat):
        x_i, y_i = feature_locations[idx]
        key = (int(x_i), int(y_i))
        if key in coord_to_featidx:
            raise RuntimeError(
                f"Duplicate coordinate {key} found at both feature‐rows "
                f"{coord_to_featidx[key]} and {idx}."
            )
        coord_to_featidx[key] = idx

    # ---- 3) Iterate through TFRecord tiles (O(M) lookups) ----
    output_patches: List[Dict] = []
    tfr_iter = sf.TFRecord(tfr_path)
    for tile_idx, rec in enumerate(tfr_iter):
        # Get exact integer coordinates (same casting rule used during feature‐save)
        x_loc = int(float(rec["loc_x"]))
        y_loc = int(float(rec["loc_y"]))
        tile_coord = (x_loc, y_loc)

        # If no feature corresponds to this tile, skip it
        if tile_coord not in coord_to_featidx:
            continue

        feat_idx = coord_to_featidx[tile_coord]
        patch_feature = features_array[feat_idx]   # shape = (D,)

        # Build an RGB‐histogram or mean‐RGB for that tile
        img = sf.io.decode_image(bytes(rec["image_raw"]))  # PIL or array
        rgb = (np.array(img).reshape(-1, 3).mean(axis=0)) / 255.0

        wsi_loc = [x_loc // patch_size, y_loc // patch_size]
        output_patches.append({
            "tfr_index": tile_idx,
            "loc":       [x_loc, y_loc],
            "wsi_loc":   wsi_loc,
            "feature":   patch_feature.copy(),  # or patch_feature itself if you don't need to modify
            "rgb_histogram": rgb.tolist()
        })

    return {
        "properties": {
            "tfr_path":            tfr_path,
            "features_index_path": features_index_path,
            "features_path":       features_path,
        },
        "patches": output_patches
    }

def save_patch_dicts_pickle(patch_data, path, compress=3):
    """
    Save patch dictionaries to disk with reduced size using joblib.

    Replaces each patch's embedded 'feature' vector with its index in the original
    feature tensor to avoid redundant storage. This makes the saved file smaller
    while retaining a way to reference the correct feature vector later.

    Args:
        patch_dicts (list): List of patch dictionaries, each potentially containing a 'feature' entry.
        path (str): Destination path for the output pickle file.
        features_tensor (torch.Tensor or np.ndarray): Tensor from which feature indices are derived.
        compress (int): Joblib compression level (0 = none, higher = more compression).
    """
    props = patch_data["properties"]
    feature_path = props.get("features_path")
    if not feature_path:
        raise ValueError("Missing 'features_path' in properties")
    feats = torch.load(feature_path)
    arr = feats.numpy() if hasattr(feats, "numpy") else feats

    simplified = []
    for patch in patch_data["patches"]:
        copy = patch.copy()
        feat = copy.pop("feature", None)
        if feat is not None:
            idxs = np.where((arr == feat).all(axis=1))[0]
            if idxs.size == 0:
                raise ValueError("Feature vector not found")
            copy["feature_index"] = int(idxs[0])
        simplified.append(copy)

    joblib.dump({"properties": props, "patches": simplified}, path, compress=compress)

def load_patch_dicts_pickle(path, reconstruct_features=False):
    """
    Load patch dictionaries from a joblib pickle and restore feature vectors.

    This function reverses the compression step done during saving by re-inserting
    the full feature vector from the given tensor using the saved 'feature_index'.

    Args:
        path (str): Path to the saved .pkl file.
    Returns:
        list: List of patch dictionaries, each containing a 'feature' vector.
    """
    data = joblib.load(path)
    if not reconstruct_features:
        # nothing to do—just return the saved dict
        return data
    props = data["properties"]
    feature_path = props.get("features_path")
    if not feature_path:
        raise ValueError("Missing 'features_path' in properties")
    feats = torch.load(feature_path)
    arr = feats.numpy() if hasattr(feats, "numpy") else feats

    recon = []
    for patch in data["patches"]:
        copy = patch.copy()
        idx = copy.pop("feature_index", None)
        if idx is not None:
            copy["feature"] = arr[idx]
        recon.append(copy)

    return {"properties": props, "patches": recon}

def save_retrieval_metrics(config, metric_results, output_path):
    """
    Save retrieval evaluation metrics to CSV in wide format with macro and micro as columns.

    Args:
        metric_results (dict): Nested dict from evaluate_retrieval_metrics.
        config (dict): Experiment config with project_name.
        save_string (str, optional): Identifier for filename suffix.
    """
    # ---- Set up results directory ----
    project = config['experiment']['project_name']
    results_dir = os.path.join('experiments', project, 'eval')
    os.makedirs(results_dir, exist_ok=True)

    # ---- Prepare rows for output DataFrame ----
    rows = []

    # Get all base metrics (ignore _macro and _micro versions)
    base_metrics = sorted({
        key for key in metric_results.keys()
        if not key.endswith('_macro') and not key.endswith('_micro')
    })

    for base in base_metrics:
        per_class = metric_results.get(base, {})  # Per-class values
        macro = metric_results.get(f"{base}_macro", {}).get('all', np.nan)  # Macro average
        micro = metric_results.get(f"{base}_micro", {}).get('all', np.nan)  # Micro average

        # Create a row with base metric name and macro/micro averages
        row = {
            'metric': base,
            'macro_average': macro,
            'micro_average': micro
        }

        # Add per-class metrics
        for label, score in per_class.items():
            row[label] = score

        rows.append(row)

    # ---- Convert to DataFrame ----
    df_wide = pd.DataFrame(rows)

    # Reorder columns so metric, macro, and micro come first
    cols = ['metric', 'macro_average', 'micro_average'] + [
        c for c in df_wide.columns if c not in ['metric', 'macro_average', 'micro_average']
    ]
    df_wide = df_wide[cols]

    # ---- Save Excel to disk ----
    df_wide.to_excel(output_path, index=False)

    logging.info(f"Saved retrieval evaluation metrics (wide) to {output_path}")

def save_retrieval_results_to_excel(results, output_path):
    """
    Converts search results into a flat DataFrame and saves it as an Excel file.

    Each row represents a query slide and its top-k retrieved results, including their
    IDs, labels, and distances. The number of top-k entries is automatically inferred.

    Args:
        results (list): Output from leave-one-patient-out with query and top_k info.
        output_path (str): Path to save the resulting Excel file.
    """
    if not results:
        raise ValueError("The results list is empty.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    k = max(len(r["top_k"]) for r in results)  # Automatically determine k from first result
    rows = []

    for result in results:
        row = {
            "Query Slide ID": result["query_slide_id"],
            "Query Label": result["query_label"],
            "Predicted Label": result["predicted_label"]
        }

        for i in range(k):
            try:
                hit = result["top_k"][i]
                row[f"Slide {i+1} ID"] = hit["slide_id"]
                row[f"Slide {i+1} Label"] = hit["label"]
                row[f"Slide {i+1} Distance"] = hit["distance"]
            except IndexError:
                # Fill with None if fewer hits than expected
                row[f"Slide {i+1} ID"] = None
                row[f"Slide {i+1} Label"] = None
                row[f"Slide {i+1} Distance"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    logging.info(f"Saved retrieval results to: {output_path}")

def log_mem(tag: str = ""):
    import os, psutil
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1e9
    cuda_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    cuda_res   = torch.cuda.memory_reserved()  / 1e6 if torch.cuda.is_available() else 0
    shm_used = 0.0
    if os.path.exists("/dev/shm"):
        try:
            shm_used = psutil.disk_usage("/dev/shm").used / 1e9
        except Exception:
            pass
    logging.info(f"[{tag}] RSS={rss:.2f} GB  CUDA alloc/res={cuda_alloc:.0f}/{cuda_res:.0f} MB  /dev/shm={shm_used:.2f} GB")

def cleanup_dev_shm():
    """Remove leaked POSIX shared memory/semaphore files created by this uid."""
    import os, stat, getpass
    base = "/dev/shm"
    if not os.path.isdir(base):
        return
    uid = os.getuid()
    for name in os.listdir(base):
        path = os.path.join(base, name)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        # only touch our own stuff
        if st.st_uid != uid:
            continue
        if name.startswith(("torch_", "pymp-", "mp.", "shm", "sem.")):
            try:
                os.unlink(path)
            except OSError:
                pass

def calculate_combinations(config: dict) -> list:
    """
    Calculate all parameter combinations based on benchmark_parameters.

    Accepts list items that are either:
      - plain strings: "method_name"
      - single-key dicts: {"method_name": {"param": value, ...}}

    Returns:
        list[dict]: One dict per combination. For any list-valued key K,
                    - if the chosen item is a string -> combo[K] = "name"
                    - if the chosen item is a dict  -> combo[K] = "name" and
                                                      combo[f"{K}_params"] = { ... }  (only if non-empty)
    """
    bp = config.get("benchmark_parameters", {})
    # Only include keys whose values are lists (same as your original behavior)
    keys = [k for k, v in bp.items() if isinstance(v, list)]

    # Build the grid of values for each key
    value_lists = []
    method_like_keys = set()
    for k in keys:
        values = bp[k]
        if not values:
            value_lists.append([None])
            continue

        # Detect if this key uses the "method" style (strings or single-key dicts)
        is_method_like = all(isinstance(x, (str, dict)) for x in values)
        if is_method_like:
            method_like_keys.add(k)

            normalized = []
            for item in values:
                if isinstance(item, str):
                    normalized.append( (item.strip(), {}) )
                elif isinstance(item, dict):
                    # Expect single-key dict: { "name": {params...} } or { "name": {} }
                    if len(item) != 1:
                        raise ValueError(f"List item for '{k}' must be a single-key mapping, got: {item}")
                    name, params = next(iter(item.items()))
                    if not isinstance(name, str):
                        raise ValueError(f"Method name for '{k}' must be a string, got: {type(name)}")
                    if params is None:
                        params = {}
                    elif not isinstance(params, dict):
                        raise ValueError(f"Params for '{k}:{name}' must be a mapping, got: {type(params)}")
                    normalized.append( (name.strip(), params) )
                else:
                    raise ValueError(f"Unsupported item type in '{k}': {type(item)}")
            value_lists.append(normalized)
        else:
            # Numeric/scalar lists (e.g., tile_px, tile_um, etc.)
            value_lists.append(values)

    # Cartesian product over all keys
    combos = []
    for choice_tuple in product(*value_lists):
        combo = {}
        for k, choice in zip(keys, choice_tuple):
            if k in method_like_keys:
                name, params = choice  # tuple(str, dict)
                combo[k] = name
                if params:  # only add when non-empty
                    combo[f"{k}_params"] = params
            else:
                combo[k] = choice
        combos.append(combo)

    return combos

def _fmt_val(v: Any) -> str:
    """Stable value formatting for IDs."""
    if isinstance(v, float):
        # Trim trailing zeros but stay readable; keep small epsilon to avoid '-0.0'
        s = f"{v:.6g}"
        return re.sub(r"(\.0+|(?<=\.\d)0+)$", "", s)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_val(x) for x in v) + "]"
    if isinstance(v, dict):
        # stable key ordering
        return "{" + ",".join(f"{k}:{_fmt_val(v[k])}" for k in sorted(v)) + "}"
    return str(v)

def _slug(s: str) -> str:
    """Make strings path-safe and compact."""
    s = str(s)
    s = s.strip().replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9._\-]", "_", s)
    return s

def _stable_hash(obj: Any, n: int = 6) -> str:
    """Short, stable hash (base16 truncated)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:n]

def _items_from_spec_and_values(spec: Dict[str, Dict[str, Any]],
                                values: Dict[str, Any]) -> list:
    """
    Convert spec+values into ordered items:
    (order, label, key, value) filtered by include_in_id.
    """
    items = []
    for key, meta in (spec or {}).items():
        if not meta.get("include_in_id", True):
            continue
        if key not in values:
            continue
        val = values[key]
        label = meta.get("id_label", key)
        order = meta.get("id_order", 1_000_000)
        items.append((order, label, key, val))
    # stable order
    items.sort(key=lambda t: (t[0], t[1]))
    return items

def build_param_id_string(
    kind: str,                # "selector" or "search"
    name: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    compact: bool = False,
    max_len: int = 120
) -> str:
    """
    Build a path-safe ID from parameter values (no heavy instantiation).

    - Reads effective values via your value helpers:
        get_mosaic_selector_values(name, params) -> dict
        get_search_method_values(name, params)   -> dict
    - Tries to fetch the class HYPERPARAMS to honor:
        include_in_id (default True), id_label, id_order
      Falls back to including all keys in alpha order if spec isn't available.
    - If compact=True -> values only in order, e.g. '10-375-375'
      else labeled, e.g. 'k10-pre375-succ375'
    """
    params = params or {}

    # 1) fetch values dict
    if kind == "selector":
        from ..image_retrieval.utils import get_mosaic_selector_values
        values = get_mosaic_selector_values(name, params)  # <-- dict only
        # best-effort fetch of spec for labels/order/inclusion
        try:
            from ..image_retrieval.mosaic_selectors.registry import get_mosaic_selector_class
            cls = get_mosaic_selector_class(name)
            spec = getattr(cls, "HYPERPARAMS", {}) or {}
        except Exception:
            spec = {}
    elif kind == "search":
        from ..image_retrieval.utils import get_search_method_values
        values = get_search_method_values(name, params)    # <-- dict only
        try:
            from ..image_retrieval.search_methods.registry import get_search_method_class
            cls = get_search_method_class(name)
            spec = getattr(cls, "HYPERPARAMS", {}) or {}
        except Exception:
            spec = {}
    else:
        raise ValueError(f"Unknown kind '{kind}' (expected 'selector' or 'search').")

    if not values:
        return ""

    # 2) collect items honoring spec when available
    items = []
    # Prefer spec order if we have it; otherwise key-sorted values
    keys_in_order = list(spec.keys()) if spec else sorted(values.keys())

    for key in keys_in_order:
        if key not in values:
            continue
        meta = spec.get(key, {})
        if not meta.get("include_in_id", True):
            continue
        label = meta.get("id_label", key)
        order = meta.get("id_order", 1_000_000)
        items.append((order, label, key, values[key]))

    # If spec was empty, we may have skipped keys not in spec; include remaining (alpha)
    if not spec:
        remaining = [k for k in sorted(values.keys()) if k not in {i[2] for i in items}]
        for k in remaining:
            items.append((1_000_000, k, k, values[k]))

    if not items:
        return ""

    # 3) stable order
    items.sort(key=lambda t: (t[0], t[1]))

    # 4) build tokens
    tokens = []
    for _, label, key, val in items:
        vs = _slug(_fmt_val(val))
        tokens.append(vs if compact else f"{_slug(label)}{vs}")

    param_id = "-".join(tokens)

    # 5) length guard
    if len(param_id) > max_len:
        payload = {k: v for _, _, k, v in items}
        h = _stable_hash(payload)
        param_id = param_id[: max_len - 8].rstrip("-") + f"--{h}"

    return param_id