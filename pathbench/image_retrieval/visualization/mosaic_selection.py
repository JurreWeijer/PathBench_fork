# Standard library
import gc
import json
import logging
import math
import os
from typing import Optional, Dict, List, Tuple

# Third-party
import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from openslide import OpenSlide
from PIL import Image, ImageDraw
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import slideflow as sf
import torch 
import matplotlib.pyplot as plt

# Local
from ...image_retrieval.utils import load_patch_dicts_pickle
from ...image_retrieval.visualization.utils import (
    get_dataset_name_for_slide,
    get_path_from_dataset,
    crop_roi,
    load_qupath_rois,
)

VIS_PATCH_SIZE = 256
MAX_THUMB_SIZE = 2048
GRID_COLS = 8
MARKER_PX = 35
PATCH_ALPHA = 0.4
LEGEND_MARGIN = 20
SWATCH_SIZE = 30
SWATCH_PAD = 10
BORDER_WIDTH = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
PATCH_VIS_METHODS = ["extensive", "simple"]

def merge_patch_polygons_by_bin(coords: np.ndarray,
                                bin_ids: np.ndarray,
                                patch_size: int):
    """
    Args:
        coords:      (N,2) array of (x, y) origins for each patch
        bin_ids:     length-N array of integer bin assignments
        patch_size:  width/height of each patch square

    Returns:
        dict[int, Polygon or MultiPolygon]:
            mapping each bin_id → the merged Polygon(s) for that bin
    """
    merged = {}

    for b in np.unique(bin_ids):
        # get indices of patches in this bin
        idxs = np.where(bin_ids == b)[0]
        
        # create a list of individual patch boxes
        boxes = []
        for i in idxs:
            x, y = coords[i]
            boxes.append(box(x, y, x + patch_size, y + patch_size))
        
        # union all boxes: adjacent boxes coalesce into bigger polygons
        unioned = unary_union(boxes)
        merged[b] = unioned

    return merged

def generate_distinct_bgr_colors(n):
    """
    Generate n visually distinct colors in BGR by sampling the HSV hue channel.
    """
    hues = np.linspace(0, 179, n, endpoint=False, dtype=int)
    colors = []
    for h in hues:
        # full saturation & value for vivid colors
        hsv = np.uint8([[[h, 255, 255]]])           # shape (1,1,3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)   # also (1,1,3)
        b, g, r = bgr[0,0].tolist()
        colors.append((b, g, r))
    return colors

def visualize_mosaic_selection_simple(
    config: dict,
    slide_id: str,
    slide_path: str,
    mosaics_folder: str
) -> Image.Image:
    """
    Render a compact visualization for a slide's *selected* patches.

    This function produces a single composite image that contains:
      1) A ROI-cropped thumbnail of the WSI (centered in a square canvas with a border).
      2) A grid containing all selected patch images below the thumbnail.
      3) Index markers drawn on the thumbnail in the approximate patch positions,
         with matching indices drawn on the grid tiles.

    Why ROI bounds matter:
      - Patches are stored in *full-resolution slide coordinates* (usually pixel
        coordinates at level 0).
      - The thumbnail returned by `crop_roi()` is a *scaled* rendering of the ROI
        (or whole slide), pasted into a square canvas.
      - To place the small black index squares in the correct spots on the
        thumbnail, we need to convert each patch location from full-resolution
        coordinates → ROI-local coordinates → thumbnail-on-canvas coordinates.

    Coordinate spaces used here:
      - Full-resolution WSI space:      (x_full, y_full) from patch dicts (level 0 px)
      - ROI-local space:                (x_full - minx, y_full - miny)
      - Thumbnail space (pre-canvas):   scale ROI-local by (thumb_img.width / W_roi,
                                         thumb_img.height / H_roi)
      - Canvas space:                   add (ox, oy) offsets because the thumbnail
                                         is centered in a MAX_THUMB_SIZE square

    Parameters
    ----------
    config : dict
        Experiment configuration; used to locate the ROI CSV path and to call `crop_roi`.
    slide_id : str
        Identifier of the slide (used for file lookup and annotations).
    slide_path : str
        Filesystem path to the WSI (e.g., .tiff, .svs).
    mosaics_folder : str
        Directory containing `{slide_id}.pkl` produced by your patch selection step.
        The PKL must include:
          - `properties.tfr_path` → path to the TFRecord
          - `patches` → list of patch dicts with keys `tfr_index` and `loc` at least

    Returns
    -------
    PIL.Image.Image
        A composite RGB image (thumbnail + patch grid + annotations).

    Notes
    -----
    - This is a *pure visualization* routine; nothing is saved to disk.
    - The function opens the TFRecord to decode patch images and makes sure to
      close it in a `finally` block (important for long batch jobs).
    - If there are no patches, the output is just the thumbnail (no grid rows).

    Raises
    ------
    FileNotFoundError
        If the ROI CSV file for this slide cannot be found by `load_qupath_rois`.
    KeyError
        If required keys are missing from the mosaic PKL (e.g., `tfr_path`).
    """

    # -------------------------------------------------------------------------
    # 0) Load ROI bounds so we can map full-res patch coords into the thumbnail
    # -------------------------------------------------------------------------
    ds_name    = get_dataset_name_for_slide(config, slide_path)
    roi_folder = get_path_from_dataset(config, ds_name, "roi_path")
    roi_csv    = os.path.join(roi_folder, f"{slide_id}.csv")

    # Read QuPath ROI geometry and extract its bounding box
    roi_geom   = load_qupath_rois(roi_csv)
    minx, miny, maxx, maxy = roi_geom.bounds
    W_roi, H_roi = maxx - minx, maxy - miny  # size of the ROI in full-res pixels

    # -------------------------------------------------------------------------
    # 1) Render a ROI-cropped thumbnail and center it on a square canvas
    # -------------------------------------------------------------------------
    # crop_roi returns:
    #   thumb_img (PIL), (W_full, H_full), scale_factor — we only need the image here.
    thumb_img, (_W_unused, _H_unused), _scale_unused = crop_roi(
        config, slide_path, slide_id, MAX_THUMB_SIZE, BORDER_WIDTH
    )

    # Create a white square canvas and paste the thumbnail in the middle.
    # ox/oy are the offsets of the pasted thumbnail inside the square canvas.
    canvas = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), (255, 255, 255))
    ox = (MAX_THUMB_SIZE - thumb_img.width) // 2
    oy = (MAX_THUMB_SIZE - thumb_img.height) // 2
    canvas.paste(thumb_img, (ox, oy))

    # Draw a simple black border around the entire square canvas.
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [BORDER_WIDTH // 2, BORDER_WIDTH // 2,
         MAX_THUMB_SIZE - BORDER_WIDTH // 2 - 1,
         MAX_THUMB_SIZE - BORDER_WIDTH // 2 - 1],
        outline="black", width=BORDER_WIDTH
    )

    # -------------------------------------------------------------------------
    # 2) Load the selected patches and decode their images from the TFRecord
    # -------------------------------------------------------------------------
    mosaic_path = os.path.join(mosaics_folder, f"{slide_id}.pkl")
    patch_data = load_patch_dicts_pickle(mosaic_path, reconstruct_features=False)

    # The TFRecord path is stored in the mosaic PKL properties by earlier steps.
    tfr_path = patch_data["properties"]["tfr_path"]
    patches  = patch_data["patches"]

    # Decode the selected patch images.
    # Use a try/finally so the TFRecord always gets closed.
    patch_images = []
    tfr = sf.TFRecord(tfr_path)
    try:
        for p in patches:
            rec = tfr[p["tfr_index"]]
            img = sf.io.decode_image(bytes(rec["image_raw"]))  # -> HxWxC array-like
            patch_images.append(Image.fromarray(np.array(img)))  # store as PIL
    finally:
        del tfr

    # -------------------------------------------------------------------------
    # 3) Build the composite canvas: thumbnail on top, patch grid below
    # -------------------------------------------------------------------------
    n_patches = len(patch_images)
    n_rows    = math.ceil(n_patches / GRID_COLS)  # how many rows of patches we need

    # Total output height = top square (thumbnail canvas) + rows of VIS_PATCH_SIZE tiles
    total_h = MAX_THUMB_SIZE + n_rows * VIS_PATCH_SIZE
    composite = Image.new("RGB", (MAX_THUMB_SIZE, total_h), (255, 255, 255))

    # Paste the centered thumbnail canvas at the top
    composite.paste(canvas, (0, 0))

    # Paste patch tiles below in a simple left-to-right grid
    for i, img in enumerate(patch_images):
        r, c = divmod(i, GRID_COLS)
        x = c * VIS_PATCH_SIZE
        y = MAX_THUMB_SIZE + r * VIS_PATCH_SIZE
        composite.paste(img, (x, y))

    # -------------------------------------------------------------------------
    # 4) Draw annotations:
    #    A) Small black index squares on the thumbnail at patch locations
    #    B) Matching indices on the grid tiles below
    # -------------------------------------------------------------------------
    # Convert the composite to OpenCV BGR for drawing text/rectangles
    cv_img = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)

    # Map ROI-local (full-res) → thumbnail pixels:
    # How many thumbnail pixels per 1 full-res pixel in ROI-space?
    scale_x = thumb_img.width  / W_roi
    scale_y = thumb_img.height / H_roi

    # A) Draw a fixed-size black square with a centered ID for each patch
    for idx, p in enumerate(patches):
        # Patch location in full-resolution WSI coordinates (level 0 px)
        x_full, y_full = p['loc']

        # Convert to ROI-local coords
        x_roi = x_full - minx
        y_roi = y_full - miny

        # Scale to thumbnail pixels, then add canvas offsets to account for centering
        dx = int(x_roi * scale_x + ox)
        dy = int(y_roi * scale_y + oy)

        # Draw a filled black square as a marker
        cv2.rectangle(cv_img, (dx, dy), (dx + MARKER_PX, dy + MARKER_PX), (0, 0, 0), -1)

        # Draw the index number centered inside the square (white text)
        text = str(idx)
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
        tx = dx + (MARKER_PX - tw) // 2
        ty = dy + (MARKER_PX + th) // 2
        cv2.putText(cv_img, text, (tx, ty), FONT, 1.0, (255, 255, 255), 2)

    # B) Label each patch in the grid with the same index (top-left corner)
    for i in range(n_patches):
        r, c = divmod(i, GRID_COLS)
        text_org = (
            c * VIS_PATCH_SIZE + 5,
            MAX_THUMB_SIZE + r * VIS_PATCH_SIZE + int(VIS_PATCH_SIZE * 0.2)
        )
        cv2.putText(cv_img, str(i), text_org, FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # 5) Convert back to PIL RGB and return
    # -------------------------------------------------------------------------
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def visualize_mosaic_selection_extensive(
    config: dict,
    slide_id: str,
    slide_path: str,
    mosaics_folder: str,
    patch_px: int,
    patch_um: float
) -> Optional[Image.Image]:

    """
    Two-panel overview: left = colored bins with a lined grid, right = selected
    patches shown as filled black squares with indices. Below: grid of patch thumbnails.
    """

    # --- 1) ROI bounds ---
    ds_name     = get_dataset_name_for_slide(config, slide_path)
    roi_folder  = get_path_from_dataset(config, ds_name, "roi_path")
    roi_csv     = os.path.join(roi_folder, f"{slide_id}.csv")

    roi_geom    = load_qupath_rois(roi_csv)
    minx, miny, maxx, maxy = roi_geom.bounds
    W, H        = maxx - minx, maxy - miny

    # --- 2) thumbnail + border via crop_roi ---
    thumb_img, (W2, H2), _scale = crop_roi(config, slide_path, slide_id, MAX_THUMB_SIZE, BORDER_WIDTH)
    tw, th      = thumb_img.size
    ox          = (MAX_THUMB_SIZE - tw) // 2
    oy          = (MAX_THUMB_SIZE - th) // 2

    canvas      = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), "white")
    canvas.paste(thumb_img, (ox, oy))
    base_cv     = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    # --- 3) full-res patch size in px ---
    slide       = OpenSlide(slide_path)
    slide_mpp   = float(slide.properties.get("openslide.mpp-x",
                         slide.properties.get("tiff_mpp_x", 1.0)))
    extraction_mpp   = patch_um / patch_px
    scale_factor     = extraction_mpp / slide_mpp
    fullres_patch_px = max(1, int(round(patch_px * scale_factor)))

    # --- mapping & exact tile size on the pasted thumbnail ---
    scale_x = tw / W
    scale_y = th / H
    thumb_patch_w = max(1, int(round(fullres_patch_px * scale_x)))
    thumb_patch_h = max(1, int(round(fullres_patch_px * scale_y)))

    def to_thumb_xy(x_fr: int, y_fr: int):
        tx = int(round((x_fr - minx) * scale_x)) + ox
        ty = int(round((y_fr - miny) * scale_y)) + oy
        return tx, ty

    # --- 4) Load groups/bin assignments & coords ---
    npz_path     = os.path.join(mosaics_folder, f"{slide_id}.npz")
    groups_path  = os.path.join(mosaics_folder, f"{slide_id}_groups.json")

    data   = np.load(npz_path)
    coords = data["coords"]  # (N, 2) full-res origins

    idx_to_bin = {}
    unique_bins = []

    if os.path.exists(groups_path):
        with open(groups_path, "r") as f:
            groups_json = json.load(f)  # { "gid": [member_idx, ...], ... }
        for gid_str, member_list in groups_json.items():
            gid = int(gid_str)
            unique_bins.append(gid)
            for i in member_list:
                idx_to_bin[int(i)] = gid
        unique_bins = sorted(unique_bins)
    else:
        if "bin_ids" not in data:
            logging.warning(f"No groups or bin_ids found for {slide_id}")
            return None
        bin_ids = data["bin_ids"].astype(int)
        unique_bins = sorted(list(np.unique(bin_ids)))
        for i, b in enumerate(bin_ids):
            idx_to_bin[int(i)] = int(b)

    # --- 5) two copies for overlays ---
    thumb_bin   = base_cv.copy()
    thumb_idx   = base_cv.copy()

    # --- 6) Left panel: colored bins + LINED GRID ---
    if unique_bins:
        overlay = thumb_bin.copy()
        palette = generate_distinct_bgr_colors(len(unique_bins))

        # Force group/bin 0 to be light gray (BGR); keep others as they are now
        LIGHT_GRAY = (200, 200, 200)  # light gray in BGR
        bin_to_color = {}
        for i, b in enumerate(unique_bins):
            if b == 0:
                bin_to_color[b] = LIGHT_GRAY
            else:
                bin_to_color[b] = palette[i]

        # Fill each tile with its bin color
        for i in range(coords.shape[0]):
            if i not in idx_to_bin:
                continue
            b = idx_to_bin[i]
            color = bin_to_color[b]

            x0_fr, y0_fr = int(coords[i][0]), int(coords[i][1])
            x0_th, y0_th = to_thumb_xy(x0_fr, y0_fr)

            cv2.rectangle(
                overlay,
                (x0_th, y0_th),
                (x0_th + thumb_patch_w - 1, y0_th + thumb_patch_h - 1),
                color,
                thickness=-1
            )

        # Blend color fill
        thumb_bin = cv2.addWeighted(overlay, PATCH_ALPHA, thumb_bin, 1 - PATCH_ALPHA, 0)

        # Draw thin black outlines to create a crisp grid look
        for i in range(coords.shape[0]):
            if i not in idx_to_bin:
                continue
            x0_fr, y0_fr = int(coords[i][0]), int(coords[i][1])
            x0_th, y0_th = to_thumb_xy(x0_fr, y0_fr)
            cv2.rectangle(
                thumb_bin,
                (x0_th, y0_th),
                (x0_th + thumb_patch_w - 1, y0_th + thumb_patch_h - 1),
                (0, 0, 0),
                thickness=1
            )

        # Legend
        max_text_width = max(
            cv2.getTextSize(f"bin {b}", FONT, 1.0, 2)[0][0]
            for b in unique_bins
        )
        legend_x = MAX_THUMB_SIZE - LEGEND_MARGIN - SWATCH_SIZE - max_text_width - 10
        total_legend_height = len(unique_bins) * SWATCH_SIZE + (len(unique_bins) - 1) * SWATCH_PAD
        legend_y = (MAX_THUMB_SIZE - total_legend_height) // 2
        for i, bin_id in enumerate(unique_bins):
            color = bin_to_color[bin_id]
            y0    = legend_y + i * (SWATCH_SIZE + SWATCH_PAD)
            cv2.rectangle(thumb_bin, (legend_x, y0),
                          (legend_x + SWATCH_SIZE, y0 + SWATCH_SIZE), color, -1)
            cv2.putText(
                thumb_bin,
                f"bin {bin_id}",
                (legend_x + SWATCH_SIZE + 5, y0 + SWATCH_SIZE - 5),
                FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA
            )
    
    # --- 7) Draw patch indices on thumb_idx & load thumbnails (reuse patch_data) ---
    mosaic_pkl_path = os.path.join(mosaics_folder, f"{slide_id}.pkl")
    patch_data = load_patch_dicts_pickle(mosaic_pkl_path, reconstruct_features=False)

    for idx, p in enumerate(patch_data["patches"]):
        x0, y0 = p["loc"]
        dx = int((x0 - minx) * scale_x + ox)
        dy = int((y0 - miny) * scale_y + oy)
        cv2.rectangle(thumb_idx, (dx, dy), (dx + MARKER_PX, dy + MARKER_PX), (0, 0, 0), -1)
        txt = str(idx)
        (w_txt, h_txt), _ = cv2.getTextSize(txt, FONT, 1.0, 2)
        tx = dx + (MARKER_PX - w_txt) // 2
        ty = dy + (MARKER_PX + h_txt) // 2
        cv2.putText(thumb_idx, txt, (tx, ty), FONT, 1.0, (255, 255, 255), 2)

    # --- 8) Load patch thumbnails for the grid below ---
    tfr = sf.TFRecord(patch_data["properties"]["tfr_path"])
    patch_imgs = []
    try:
        for p in patch_data["patches"]:
            rec = tfr[p["tfr_index"]]
            img = sf.io.decode_image(bytes(rec["image_raw"]))
            patch_imgs.append(Image.fromarray(np.array(img)))
    finally:
        del tfr

    # --- 9) Composite two thumbs + full-width grid below ---
    cols   = GRID_COLS * 2
    n      = len(patch_imgs)
    rows   = math.ceil(n / cols)
    H_tot  = MAX_THUMB_SIZE + rows * VIS_PATCH_SIZE

    comp = Image.new("RGB", (2 * MAX_THUMB_SIZE, H_tot), "white")
    comp.paste(Image.fromarray(cv2.cvtColor(thumb_bin, cv2.COLOR_BGR2RGB)), (0, 0))
    comp.paste(Image.fromarray(cv2.cvtColor(thumb_idx, cv2.COLOR_BGR2RGB)), (MAX_THUMB_SIZE, 0))

    for i, img in enumerate(patch_imgs):
        r, c = divmod(i, cols)
        comp.paste(img, (c * VIS_PATCH_SIZE, MAX_THUMB_SIZE + r * VIS_PATCH_SIZE))

    # --- 10) Label the grid patches below ---
    cvc = cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2BGR)
    for i in range(n):
        r, c = divmod(i, cols)
        x = c * VIS_PATCH_SIZE + 5
        y = MAX_THUMB_SIZE + r * VIS_PATCH_SIZE + int(VIS_PATCH_SIZE * 0.2)
        cv2.putText(cvc, str(i), (x, y), FONT, 1.0, (0, 0, 0), 2)

    return Image.fromarray(cv2.cvtColor(cvc, cv2.COLOR_BGR2RGB))

"""def visualize_mosaic_selection_extensive(
    config: dict,
    slide_id: str,
    slide_path: str,
    mosaics_folder: str,
    patch_px: int,
    patch_um: float
) -> Optional[Image.Image]:
    
    # --- 1) ROI bounds ---
    ds_name     = get_dataset_name_for_slide(config, slide_path)
    roi_folder  = get_path_from_dataset(config, ds_name, "roi_path")
    roi_csv     = os.path.join(roi_folder, f"{slide_id}.csv")

    roi_geom    = load_qupath_rois(roi_csv)
    minx, miny, maxx, maxy = roi_geom.bounds
    W, H        = maxx - minx, maxy - miny

    # --- 2) thumbnail + border via crop_roi ---
    thumb_img, (W2, H2), _scale = crop_roi(config, slide_path, slide_id, MAX_THUMB_SIZE, BORDER_WIDTH)
    tw, th      = thumb_img.size
    ox          = (MAX_THUMB_SIZE - tw) // 2
    oy          = (MAX_THUMB_SIZE - th) // 2

    canvas      = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), "white")
    canvas.paste(thumb_img, (ox, oy))
    base_cv     = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    # --- 3) true full-res patch size in px ---
    slide       = OpenSlide(slide_path)
    slide_mpp   = float(slide.properties.get(
                        "openslide.mpp-x",
                        slide.properties.get("tiff_mpp_x", 1.0)
                    ))
    extraction_mpp   = patch_um / patch_px
    scale_factor     = extraction_mpp / slide_mpp
    fullres_patch_px = max(1, int(round(patch_px * scale_factor)))

    # --- 4) Load & merge groups (prefer groups.json; fallback to NPZ) ---
    npz_path     = os.path.join(mosaics_folder, f"{slide_id}.npz")
    groups_path  = os.path.join(mosaics_folder, f"{slide_id}_groups.json")

    # coords always needed to build boxes
    data   = np.load(npz_path)
    coords = data["coords"]  # (N, 2) full-res origins

    merged = {}

    if os.path.exists(groups_path):
        with open(groups_path, "r") as f:
            groups_json = json.load(f)  # { "gid": [member_idx, ...], ... }
        for gid_str, member_list in groups_json.items():
            gid = int(gid_str)  # robust to string keys
            boxes = []
            for i in member_list:
                x0, y0 = int(coords[i][0]), int(coords[i][1])
                boxes.append(Polygon([
                    (x0,                    y0),
                    (x0 + fullres_patch_px, y0),
                    (x0 + fullres_patch_px, y0 + fullres_patch_px),
                    (x0,                    y0 + fullres_patch_px),
                ]))
            merged[gid] = unary_union(boxes)
    else:
        bin_ids = data["bin_ids"]
        for b in np.unique(bin_ids):
            idxs = np.where(bin_ids == b)[0]
            boxes = []
            for i in idxs:
                x0, y0 = int(coords[i][0]), int(coords[i][1])
                boxes.append(Polygon([
                    (x0,                    y0),
                    (x0 + fullres_patch_px, y0),
                    (x0 + fullres_patch_px, y0 + fullres_patch_px),
                    (x0,                    y0 + fullres_patch_px),
                ]))
            merged[b] = unary_union(boxes)

    # --- 5) two copies for bin/index overlays ---
    thumb_bin   = base_cv.copy()
    thumb_idx   = base_cv.copy()

    # mapping full-res → thumb
    scale_x     = tw / W
    scale_y     = th / H

    # --- 6) Draw colored group overlays (with empty-guard) ---
    unique_bins = sorted(merged.keys())
    if unique_bins:
        overlay = thumb_bin.copy()
        palette = generate_distinct_bgr_colors(len(unique_bins))
        for i, b in enumerate(unique_bins):
            color = palette[i]
            poly  = merged[b]
            parts = [poly] if isinstance(poly, Polygon) else poly.geoms
            for pg in parts:
                pts = np.array([
                    [int((x - minx) * scale_x + ox),
                     int((y - miny) * scale_y + oy)]
                    for (x, y) in pg.exterior.coords
                ], np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, color, 2)
        thumb_bin = cv2.addWeighted(overlay, PATCH_ALPHA, thumb_bin, 1 - PATCH_ALPHA, 0)

        # Legend (only if we have bins)
        max_text_width = max(
            cv2.getTextSize(f"bin {b}", FONT, 1.0, 2)[0][0]
            for b in unique_bins
        )
        legend_x = MAX_THUMB_SIZE - LEGEND_MARGIN - SWATCH_SIZE - max_text_width - 10
        total_legend_height = len(unique_bins) * SWATCH_SIZE + (len(unique_bins) - 1) * SWATCH_PAD
        legend_y = (MAX_THUMB_SIZE - total_legend_height) // 2
        for i, bin_id in enumerate(unique_bins):
            color = palette[i]
            y0    = legend_y + i * (SWATCH_SIZE + SWATCH_PAD)
            # swatch (square)
            cv2.rectangle(thumb_bin, (legend_x, y0),
                          (legend_x + SWATCH_SIZE, y0 + SWATCH_SIZE), color, -1)
            # label
            cv2.putText(
                thumb_bin,
                f"bin {bin_id}",
                (legend_x + SWATCH_SIZE + 5, y0 + SWATCH_SIZE - 5),
                FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA
            )
    # else: nothing to draw; keep plain thumb_bin

    # --- 7) Draw patch indices on thumb_idx & load thumbnails (reuse patch_data) ---
    mosaic_pkl_path = os.path.join(mosaics_folder, f"{slide_id}.pkl")
    patch_data = load_patch_dicts_pickle(mosaic_pkl_path, reconstruct_features=False)

    for idx, p in enumerate(patch_data["patches"]):
        x0, y0 = p["loc"]
        dx = int((x0 - minx) * scale_x + ox)
        dy = int((y0 - miny) * scale_y + oy)
        cv2.rectangle(thumb_idx, (dx, dy), (dx + MARKER_PX, dy + MARKER_PX), (0, 0, 0), -1)
        txt = str(idx)
        (w_txt, h_txt), _ = cv2.getTextSize(txt, FONT, 1.0, 2)
        tx = dx + (MARKER_PX - w_txt) // 2
        ty = dy + (MARKER_PX + h_txt) // 2
        cv2.putText(thumb_idx, txt, (tx, ty), FONT, 1.0, (255, 255, 255), 2)

    # --- 8) Load patch thumbnails for the grid below (and close TFRecord) ---
    tfr = sf.TFRecord(patch_data["properties"]["tfr_path"])
    patch_imgs = []
    try:
        for p in patch_data["patches"]:
            rec = tfr[p["tfr_index"]]
            img = sf.io.decode_image(bytes(rec["image_raw"]))
            patch_imgs.append(Image.fromarray(np.array(img)))
    finally:
        del tfr

    # --- 9) Composite two thumbs + full-width grid below ---
    cols   = GRID_COLS * 2
    n      = len(patch_imgs)
    rows   = math.ceil(n / cols)
    H_tot  = MAX_THUMB_SIZE + rows * VIS_PATCH_SIZE

    comp = Image.new("RGB", (2 * MAX_THUMB_SIZE, H_tot), "white")
    comp.paste(Image.fromarray(cv2.cvtColor(thumb_bin, cv2.COLOR_BGR2RGB)), (0, 0))
    comp.paste(Image.fromarray(cv2.cvtColor(thumb_idx, cv2.COLOR_BGR2RGB)), (MAX_THUMB_SIZE, 0))

    for i, img in enumerate(patch_imgs):
        r, c = divmod(i, cols)
        comp.paste(img, (c * VIS_PATCH_SIZE, MAX_THUMB_SIZE + r * VIS_PATCH_SIZE))

    # --- 10) Label the grid patches below (fix color space conversion) ---
    cvc = cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2BGR)  # correct: PIL RGB -> OpenCV BGR
    for i in range(n):
        r, c = divmod(i, cols)
        x = c * VIS_PATCH_SIZE + 5
        y = MAX_THUMB_SIZE + r * VIS_PATCH_SIZE + int(VIS_PATCH_SIZE * 0.2)
        cv2.putText(cvc, str(i), (x, y), FONT, 1.0, (0, 0, 0), 2)

    # Convert back to PIL RGB for return
    return Image.fromarray(cv2.cvtColor(cvc, cv2.COLOR_BGR2RGB))"""

def add_title(img: Image.Image, title: str, bar_height: int = 150,
                 font_scale: float = 3.0, thickness: int = 4) -> Image.Image:
    """
    Add a title bar above a PIL image by using OpenCV putText.
    No external font file needed—just OpenCV’s HersheySimplex.
    """
    # 1) Convert PIL→OpenCV
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h_img, w_img = arr.shape[:2]

    # 2) Make a new canvas with extra bar_height at top
    canvas = np.full((h_img + bar_height, w_img, 3), 255, dtype=np.uint8)

    # 3) Paste the image into the bottom
    canvas[bar_height:] = arr

    # 4) Measure text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, thickness)

    # 5) Compute center position
    x = (w_img - text_w) // 2
    # y at roughly middle of the bar; OpenCV’s y is baseline of text
    y = (bar_height + text_h) // 2

    # 6) Draw text in black
    cv2.putText(canvas, title, (x, y), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

    # 7) Convert back to PIL
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def generate_mosaic_selection_report_pdf(
    config,
    all_data,
    slide_mosaic_paths,
    mosaic_selector,
    pdf_base,
    patch_px,
    patch_um
):
    """
    Generate patch selection visualizations (simple or extensive) 
    and export them to multi-page PDFs.

    Configuration is taken from the `patch_cfg` dictionary, which should include:
      - "mode": visualization style, "simple" or "extensive".
      - "max_per_file" (optional): maximum slides per PDF (default 200).

    Args:
        patch_cfg (dict): Settings from config["visualization"]["patch_selection"].
        all_data (sf.Project): Project for locating slides.
        slide_mosaic_paths (dict): Mapping slide_id → mosaic pickle path.
        mosaic_selector (str): Name of the patch selection method.
        pdf_base (str): Base filename for output PDFs.
        patch_px (int): Patch size in pixels.
        patch_um (float): Patch size in microns.
    """
    # detect which mode to use
    patch_cfg = config['experiment'].get("visualization", {}).get("patch_selection", {})
    mode = patch_cfg.get("mode", "simple")
    max_per_file = patch_cfg.get("max_per_file", 200)

    if mode not in ("simple", "extensive"):
        raise ValueError(f"Invalid patch_selection mode '{mode}'. Must be 'simple' or 'extensive'.")

    logging.info(f"Generating patch selection report in '{mode}' mode")

    items = list(slide_mosaic_paths.items())
    total = len(items)
    parts = math.ceil(total / max_per_file)

    for part in range(parts):
        start = part * max_per_file
        end   = min(start + max_per_file, total)
        chunk = items[start:end]

        out_pdf = f"{pdf_base}.pdf" if parts == 1 else f"{pdf_base}_part{part+1}.pdf"
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

        with PdfPages(out_pdf) as pdf:
            for slide_id, mosaic_pkl in chunk:
                fig = None            # <-- initialize so it's always defined
                img = None
                arr = None
                try:
                    slide_path = all_data.find_slide(slide=slide_id)
                    if slide_path is None:
                        logging.warning(f"Slide not found: {slide_id}")
                        continue

                    mosaics_folder = os.path.dirname(mosaic_pkl)

                    # ---- generate 1 image and immediately write it ----
                    if mode == "simple":
                        img = visualize_mosaic_selection_simple(
                            config, slide_id, slide_path, mosaics_folder
                        )
                    else:
                        img = visualize_mosaic_selection_extensive(
                            config, slide_id, slide_path, mosaics_folder,
                            patch_px, patch_um
                        )
                    if img is None:
                        continue

                    img = add_title(img, f"{slide_id} - {mosaic_selector}")
                    arr = np.array(img)

                    fig = plt.figure(figsize=(arr.shape[1]/100, arr.shape[0]/100), dpi=100)
                    plt.axis("off")
                    plt.imshow(arr)
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.0)
                    plt.close(fig)

                except Exception as e:
                    logging.warning(f"Viz failed for {slide_id}: {e}")

                finally:
                    if fig is not None:
                        plt.close(fig)
                    # break reference cycles ASAP
                    img = None
                    arr = None
                    gc.collect()

        logging.info(f"Wrote slides {start+1}-{end} → {out_pdf}")
    
    # ---- FINAL CLEANUP -------------------------------------------------
    try:
        img  # type: ignore
        arr  # type: ignore
        items  # type: ignore
        chunk  # type: ignore
        mosaics_folder  # type: ignore
        slide_mosaic_paths  # type: ignore
        all_data  # type: ignore
    except NameError:
        pass
    else:
        img = arr = items = chunk = None
        mosaics_folder = slide_mosaic_paths = all_data = None

    # Close any stray matplotlib figures (PdfPages context already closed)
    plt.close('all')

    gc.collect()
    if 'torch' in globals() and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return