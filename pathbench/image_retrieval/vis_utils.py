from typing import Dict
import os 
from openslide import OpenSlide
import joblib
from PIL import Image
import torch
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
import pandas as pd 

def get_dataset_name_for_slide(config: Dict, slide_filepath: str) -> str:
    """
    Given an absolute or relative slide file path and a config dict (already loaded),
    return the dataset 'name' whose 'slide_path' is a parent of the slide_filepath.

    Args:
        slide_filepath: Path to the slide file.
        config:         Configuration dict containing a 'datasets' list.

    Returns:
        The dataset name (str).

    Raises:
        ValueError if no matching dataset is found.
    """
    slide_abspath = os.path.abspath(slide_filepath)

    for ds in config.get("datasets", []):
        ds_slide_dir = os.path.abspath(ds.get("slide_path", ""))
        # check if slide_abspath is inside ds_slide_dir
        try:
            common = os.path.commonpath([slide_abspath, ds_slide_dir])
        except ValueError:
            continue  # on windows if different drives, skip
        if common == ds_slide_dir:
            return ds["name"]

    raise ValueError(f"No dataset in config contains slide: {slide_filepath}")

def get_path_from_dataset(config: Dict, dataset_name: str, path_name: str) -> str:
    """
    Given a dataset name and the loaded config dict, return that dataset’s roi_path.

    Args:
        dataset_name: The 'name' field of the dataset you’re interested in.
        config:       The configuration dict containing the 'datasets' list.

    Returns:
        The roi_path string for the matching dataset.

    Raises:
        KeyError if the dataset name isn't found or if 'roi_path' is missing.
    """
    for ds in config.get("datasets", []):
        if ds.get("name") == dataset_name:
            path = ds.get(path_name)
            if path:
                return os.path.abspath(path)
            else:
                return None

def crop_roi(config, slide_path, slide_id, thumb_size, border_px):
    """
    Crop the tissue ROI (if defined) or the full slide into a square thumbnail
    of side THUMB_SIZE (minus borders), preserving aspect ratio.

    Returns:
        img (PIL.Image): The cropped & resized image.
        full_size (tuple): (W, H) full‐resolution dimensions of the crop region.
        scale (float): Scale factor from level‐coords → thumbnail inner size.
    """
    sl = OpenSlide(slide_path)

    # 1) Try to get an ROI folder; if none, we'll fall back to full slide
    try:
        ds_name = get_dataset_name_for_slide(config, slide_path)
        roi_fld = get_path_from_dataset(config, ds_name, "roi_path")
    except (KeyError, TypeError):
        roi_fld = None

    # 2) Determine crop bounds
    if roi_fld:
        csv_path = os.path.join(roi_fld, f"{slide_id}.csv")
        if os.path.exists(csv_path):
            geom = load_qupath_rois(csv_path)
            minx, miny, maxx, maxy = geom.bounds
        else:
            # no CSV for this slide → full slide
            w0, h0 = sl.level_dimensions[0]
            minx, miny, maxx, maxy = 0, 0, w0, h0
    else:
        # no ROI folder configured → full slide
        w0, h0 = sl.level_dimensions[0]
        minx, miny, maxx, maxy = 0, 0, w0, h0

    # 3) Pick best level so that the crop fits within the THUMB_SIZE (including borders)
    lvl, _   = find_best_level(sl, thumb_size)
    down     = sl.level_downsamples[lvl]

    # full‐res width/height of the region
    W, H     = maxx - minx, maxy - miny
    # level‐coords size
    w_l, h_l = int(W / down), int(H / down)

    # 4) Read that region
    img = sl.read_region((int(minx), int(miny)), lvl, (w_l, h_l)).convert("RGB")

    # 5) Resize to fit within the inner square (accounting for a BORDER_PX on each side)
    inner = thumb_size - 2 * border_px
    scale = min(inner / w_l, inner / h_l)
    tw, th = int(w_l * scale), int(h_l * scale)
    img = img.resize((tw, th), Image.BILINEAR)

    return img, (W, H), scale

def find_best_level(slide: OpenSlide, max_size: int = 2048):
    """
    Return the (level_index, downsample) of the highest-resolution
    level whose dimensions both do not exceed max_thumb_size.
    If none, returns the bottom‐most (smallest) level.
    """
    # slide.level_dimensions is a tuple of (w, h) for each level
    # slide.level_downsamples is the corresponding downsample factor
    for lvl, (w, h) in enumerate(slide.level_dimensions):
        if w <= max_size and h <= max_size:
            return lvl, slide.level_downsamples[lvl]
    # fallback to the smallest (last) level
    return slide.level_count - 1, slide.level_downsamples[-1]

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

def load_qupath_rois(csv_path: str) -> MultiPolygon:
    """
    Load QuPath‐exported CSV with columns [ROI_Name, X_base, Y_base]
    and return a unified Shapely polygon (possibly MultiPolygon).
    """
    df = pd.read_csv(csv_path)
    polys = []
    for name, grp in df.groupby("ROI_Name"):
        coords = list(zip(grp["X_base"], grp["Y_base"]))
        polys.append(Polygon(coords))
    return unary_union(polys)
