from openslide import OpenSlide
import slideflow as sf
import math
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
import random
import cv2
import joblib
import numpy as np
from shapely.geometry import polygon
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import math 
import slideflow as sf
import cv2
from typing import Dict, Optional
import gc

from ..image_retrieval.utils import load_patch_dicts_pickle
from .vis_utils import get_dataset_name_for_slide, get_path_from_dataset, crop_roi, load_qupath_rois, load_patch_dicts_pickle

VIS_PATCH_SIZE = 256
MAX_THUMB_SIZE = 2048
GRID_COLS = 8
MARKER_PX = 35
PATCH_ALPHA = 0.4
LEGEND_MARGING = 20
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

def visualize_selected_patches_simple(
    config: dict,
    slide_id: str,
    slide_path: str,
    mosaics_folder: str
) -> Image.Image:
    """
    Simple patch selection visualization, refactored to use crop_roi.

    Args:
        config: experiment config dict (for ROI lookup)
        slide_id: slide identifier
        slide_path: path to .tiff file
        mosaics_folder: folder containing slide_id.pkl

    Returns:
        composite PIL Image
    """
    # 1) Crop and resize ROI (or full slide) into a thumbnail
    thumb_img, (W, H), scale = crop_roi(config, slide_path, slide_id, MAX_THUMB_SIZE, BORDER_WIDTH)

    # 2) Paste into square canvas with border
    canvas = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), (255, 255, 255))
    ox = (MAX_THUMB_SIZE - thumb_img.width) // 2
    oy = (MAX_THUMB_SIZE - thumb_img.height) // 2
    canvas.paste(thumb_img, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [BORDER_WIDTH//2, BORDER_WIDTH//2,
         MAX_THUMB_SIZE - BORDER_WIDTH//2 - 1,
         MAX_THUMB_SIZE - BORDER_WIDTH//2 - 1],
        outline="black", width=BORDER_WIDTH
    )

    # 3) Load patches
    mosaic_path = os.path.join(mosaics_folder, f"{slide_id}.pkl")
    patch_data = load_patch_dicts_pickle(mosaic_path, reconstruct_features=False)
    tfr = sf.TFRecord(patch_data["properties"]["tfr_path"])
    patches = patch_data["patches"]

    patch_images = []
    for p in patches:
        rec = tfr[p["tfr_index"]]
        img = sf.io.decode_image(bytes(rec["image_raw"]))
        patch_images.append(Image.fromarray(np.array(img)))
    tfr.close()                
    del tfr

    # 4) Build composite grid below thumbnail
    n = len(patch_images)
    rows = math.ceil(n / GRID_COLS)
    total_h = MAX_THUMB_SIZE + rows * VIS_PATCH_SIZE

    composite = Image.new("RGB", (MAX_THUMB_SIZE, total_h), (255,255,255))
    composite.paste(canvas, (0, 0))
    for i, img in enumerate(patch_images):
        r, c = divmod(i, GRID_COLS)
        composite.paste(img, (c * VIS_PATCH_SIZE, MAX_THUMB_SIZE + r * VIS_PATCH_SIZE))

    # 5) Annotate ROI thumbnail with patch markers
    cv_img = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
    scale_x = thumb_img.width / W
    scale_y = thumb_img.height / H

    for idx, p in enumerate(patches):
        x_full, y_full = p['loc']
        dx = int((x_full - 0) * scale_x + ox)
        dy = int((y_full - 0) * scale_y + oy)
        cv2.rectangle(cv_img, (dx, dy), (dx+MARKER_PX, dy+MARKER_PX), (0,0,0), -1)
        text = str(idx)
        (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 2)
        tx = dx + (MARKER_PX - tw)//2
        ty = dy + (MARKER_PX + th)//2
        cv2.putText(cv_img, text, (tx, ty), FONT, 1.0, (255,255,255), 2)

    # 6) Annotate grid patches
    for i in range(n):
        r, c = divmod(i, GRID_COLS)
        org = (c * VIS_PATCH_SIZE + 5,
               MAX_THUMB_SIZE + r * VIS_PATCH_SIZE + int(VIS_PATCH_SIZE * 0.2))
        cv2.putText(cv_img, str(i), org, FONT, 1.0, (0,0,0), 2, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def visualize_selected_patches_extensive(
    config: dict,
    slide_id: str,
    slide_path: str,
    mosaics_folder: str,
    patch_px: int,
    patch_um: float
) -> Optional[Image.Image]:
    """
    Two-panel overview: left = colored bins, right = numbered patches,
    both over the same ROI-cropped thumbnail (with true patch-size overlays).
    Below: full-width grid of patch thumbnails.

    Args:
        config:         your config dict.
        slide_id:       ID of the slide.
        slide_path:     Filesystem path to the WSI.
        mosaics_folder: Dir containing {slide_id}.pkl + .npz.
        patch_px:       Tile side length (px) AT THE EXTRACTION MPP.
        patch_um:       Microns per side of patch (so patch MPP = patch_um/patch_px).
    """
    # --- 1) Figure out where the ROI CSV lives, load its bounds ---
    ds_name     = get_dataset_name_for_slide(config, slide_path)
    roi_folder  = get_path_from_dataset(config, ds_name, "roi_path")
    roi_csv     = os.path.join(roi_folder, f"{slide_id}.csv")

    roi_geom    = load_qupath_rois(roi_csv)
    minx, miny, maxx, maxy = roi_geom.bounds
    W, H        = maxx - minx, maxy - miny

    # --- 2) Build a centered thumbnail + border via your crop_roi helper ---
    thumb_img, (W2, H2), _scale = crop_roi(config, slide_path, slide_id, MAX_THUMB_SIZE, BORDER_WIDTH)
    tw, th      = thumb_img.size
    ox          = (MAX_THUMB_SIZE - tw) // 2
    oy          = (MAX_THUMB_SIZE - th) // 2

    canvas      = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), "white")
    canvas.paste(thumb_img, (ox, oy))
    base_cv     = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    # --- 3) Compute true full-res patch size in px ---
    slide       = OpenSlide(slide_path)
    slide_mpp   = float(slide.properties.get(
                       "openslide.mpp-x",
                       slide.properties.get("tiff_mpp_x", 1.0)
                   ))
    extraction_mpp   = patch_um / patch_px
    scale_factor     = extraction_mpp / slide_mpp
    fullres_patch_px = max(1, int(round(patch_px * scale_factor)))

    # --- 4) Load & merge the bin polygons in full-res coords ---
    npz_path    = os.path.join(mosaics_folder, f"{slide_id}.npz")
    data        = np.load(npz_path)
    bin_ids     = data["bin_ids"]
    coords      = data["coords"]

    merged = {}
    for b in np.unique(bin_ids):
        idxs = np.where(bin_ids == b)[0]
        boxes = []
        for i in idxs:
            x0, y0 = coords[i]
            boxes.append(Polygon([
                (x0, y0),
                (x0 + fullres_patch_px, y0),
                (x0 + fullres_patch_px, y0 + fullres_patch_px),
                (x0, y0 + fullres_patch_px),
            ]))
        merged[b] = unary_union(boxes)

    # --- 5) Prepare two copies for bin & index overlays ---
    thumb_bin   = base_cv.copy()
    thumb_idx   = base_cv.copy()

    # mapping factors full-res→thumb
    scale_x     = tw / W
    scale_y     = th / H

    # --- 6) Draw colored bins on thumb_bin ---
    overlay     = thumb_bin.copy()
    unique_bins = sorted(merged.keys())
    palette     = generate_distinct_bgr_colors(len(unique_bins))
    for i, b in enumerate(unique_bins):
        color = palette[i]
        poly  = merged[b]
        parts = [poly] if isinstance(poly, Polygon) else poly.geoms
        for pg in parts:
            pts = np.array([
                [int((x-minx)*scale_x + ox),
                 int((y-miny)*scale_y + oy)]
                for (x, y) in pg.exterior.coords
            ], np.int32).reshape(-1,1,2)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 2)
    thumb_bin = cv2.addWeighted(overlay, PATCH_ALPHA, thumb_bin, 1-PATCH_ALPHA, 0)

    # --- ** LEGEND back in here ** ---
    # compute text width for safety
    max_text_width = max(
        cv2.getTextSize(f"bin {b}", FONT, 1.0, 2)[0][0]
        for b in unique_bins
    )
    # right-side, vertically centered
    legend_x = MAX_THUMB_SIZE - LEGEND_MARGING - SWATCH_SIZE - max_text_width - 10
    total_legend_height = len(unique_bins)*SWATCH_SIZE + (len(unique_bins)-1)*SWATCH_PAD
    legend_y = (MAX_THUMB_SIZE - total_legend_height) // 2
    for i, bin_id in enumerate(unique_bins):
        color = palette[i]
        y0    = legend_y + i*(SWATCH_SIZE + SWATCH_PAD)
        # swatch
        cv2.rectangle(
            thumb_bin,
            (legend_x, y0),
            (legend_x + SWATCH_SIZE, y0 + SWATCH_PAD),
            color, -1
        )
        # label
        cv2.putText(
            thumb_bin,
            f"bin {bin_id}",
            (legend_x + SWATCH_SIZE + 5, y0 + SWATCH_SIZE - 5),
            FONT, 1.0, (0,0,0), 2, cv2.LINE_AA
        )

    # --- 7) Draw patch indices on thumb_idx ---
    patch_data = load_patch_dicts_pickle(
        os.path.join(mosaics_folder, f"{slide_id}.pkl"),
        reconstruct_features=False
    )
    for idx, p in enumerate(patch_data["patches"]):
        x0, y0 = p["loc"]
        dx = int((x0-minx)*scale_x + ox)
        dy = int((y0-miny)*scale_y + oy)
        cv2.rectangle(thumb_idx, (dx,dy),
                      (dx+MARKER_PX,dy+MARKER_PX),
                      (0,0,0), -1)
        txt = str(idx)
        (w_txt,h_txt),_ = cv2.getTextSize(txt, FONT, 1.0, 2)
        tx = dx + (MARKER_PX - w_txt)//2
        ty = dy + (MARKER_PX + h_txt)//2
        cv2.putText(thumb_idx, txt, (tx,ty),
                    FONT, 1.0, (255,255,255), 2)

    # --- 8) Load patch thumbnails for the grid below ---
    tfr = sf.TFRecord(patch_data["properties"]["tfr_path"])
    patch_imgs = []
    for p in patch_data["patches"]:
        rec = tfr[p["tfr_index"]]
        img = sf.io.decode_image(bytes(rec["image_raw"]))
        patch_imgs.append(Image.fromarray(np.array(img)))

    # --- 9) Composite the two thumbs + full-width grid below ---
    cols   = GRID_COLS * 2
    n      = len(patch_imgs)
    rows   = math.ceil(n / cols)
    H_tot  = MAX_THUMB_SIZE + rows * VIS_PATCH_SIZE

    comp = Image.new("RGB", (2*MAX_THUMB_SIZE, H_tot), "white")
    comp.paste(Image.fromarray(cv2.cvtColor(thumb_bin, cv2.COLOR_BGR2RGB)), (0,0))
    comp.paste(Image.fromarray(cv2.cvtColor(thumb_idx, cv2.COLOR_BGR2RGB)), (MAX_THUMB_SIZE,0))

    for i, img in enumerate(patch_imgs):
        r, c = divmod(i, cols)
        comp.paste(img, (c*VIS_PATCH_SIZE, MAX_THUMB_SIZE + r*VIS_PATCH_SIZE))

    # --- 10) Label the grid patches below ---
    cvc = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)
    for i in range(n):
        r, c = divmod(i, cols)
        x = c*VIS_PATCH_SIZE + 5
        y = MAX_THUMB_SIZE + r*VIS_PATCH_SIZE + int(VIS_PATCH_SIZE*0.2)
        cv2.putText(cvc, str(i), (x,y), FONT, 1.0, (0,0,0), 2)

    return Image.fromarray(cvc)

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

def generate_patch_selection_report_pdf(
    config,
    all_data,
    slide_mosaic_paths,
    mosaic_method,
    pdf_base,
    patch_px,
    patch_um, 
    max_per_file: int = 200
):
    """
    Iterate over every slide_id → mosaic pickle, generate each visualization
    (simple or extensive) and write them as pages in one or more PDFs,
    using Pillow’s multi‐page PDF save.

    Args:
        config:             Experiment configuration dict.
        all_data:           slideflow.Project or Dataset for locating slides.
        slide_mosaic_paths: Mapping slide_id → path to mosaic .pkl.
        mosaic_method:      Name of the patch‐selection method.
        pdf_base:           Base path (no extension) for output PDF(s).
                            E.g. "/path/to/reports/patch_report"
        max_per_file:       Maximum slides per PDF.
    """
    # detect which mode to use
    vizs = config["experiment"].get("visualization", [])
    patch_vis = [v for v in vizs if v.startswith("patch_selection-")]
    mode = (patch_vis[0].split("-",1)[1] if patch_vis else "simple")
    if mode not in ("simple","extensive"):
        mode = "simple"

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
                try:
                    slide_path = all_data.find_slide(slide=slide_id)
                    if slide_path is None:
                        logging.warning(f"Slide not found: {slide_id}")
                        continue

                    mosaics_folder = os.path.dirname(mosaic_pkl)

                    # ---- generate 1 image and immediately write it ----
                    if mode == "simple":
                        img = visualize_selected_patches_simple(
                            config, slide_id, slide_path, mosaics_folder
                        )
                    else:
                        img = visualize_selected_patches_extensive(
                            config, slide_id, slide_path, mosaics_folder,
                            patch_px, patch_um
                        )
                    if img is None:
                        continue

                    img = add_title(img, f"{slide_id} - {mosaic_method}")
                    arr = np.array(img)

                    fig = plt.figure(figsize=(arr.shape[1]/100, arr.shape[0]/100), dpi=100)
                    plt.axis("off")
                    plt.imshow(arr)
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.0)
                    plt.close(fig)

                except Exception as e:
                    logging.warning(f"Viz failed for {slide_id}: {e}")

                # ---- FREE per-slide memory ----
                del img, arr
                gc.collect()
        logging.info(f"Wrote slides {start+1}-{end} → {out_pdf}")
    
    # ---- FINAL CLEANUP -------------------------------------------------
    import gc, psutil, os
    try:
        import torch
    except ImportError:
        torch = None

    # Drop big locals that may still reference images/arrays
    for name in ["items", "chunk", "img", "arr", "slide_mosaic_paths",
                 "all_data", "mosaics_folder"]:
        if name in locals():
            del locals()[name]

    # Close any stray matplotlib figures (PdfPages context is already closed)
    import matplotlib.pyplot as plt
    plt.close('all')

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return

"""def generate_patch_selection_report_pdf(
    config,
    all_data,
    slide_mosaic_paths,
    mosaic_method,
    pdf_base,
    patch_px,
    patch_um, 
    max_per_file: int = 200
):
    # --- collect all composites up front ---
    composites = []
    slide_ids  = []
    # detect which mode to use
    vizs      = config["experiment"].get("visualization", [])
    patch_vis = [v for v in vizs if v.startswith("patch_selection-")]
    if not patch_vis:
        logging.info("No patch_selection method specified; defaulting to 'simple'")
        mode = "simple"
    else:
        mode = patch_vis[0].split("-",1)[1]
        if mode not in ("simple","extensive"):
            logging.warning(f"Unknown patch_selection mode '{mode}', defaulting to simple")
            mode = "simple"

    for slide_id, mosaic_pkl in slide_mosaic_paths.items():
        slide_path = all_data.find_slide(slide=slide_id)
        if slide_path is None:
            logging.warning(f"Slide not found: {slide_id}")
            continue

        mosaics_folder = os.path.dirname(mosaic_pkl)

        # generate the composite PIL image
        if mode == "simple":
            img = visualize_selected_patches_simple(
                config=config,
                slide_id=slide_id,
                slide_path=slide_path,
                mosaics_folder=mosaics_folder,
            )
        else:
            img = visualize_selected_patches_extensive(
                config=config,
                slide_id=slide_id,
                slide_path=slide_path,
                mosaics_folder=mosaics_folder, 
                patch_px=patch_px,
                patch_um=patch_um
            )
        if img is None:
            logging.warning(f"Visualization failed for {slide_id}")
            continue

        composites.append(img.convert("RGB"))
        slide_ids.append(slide_id)

    titled_composites = []
    for img, slide_id in zip(composites, slide_ids):
        title = f"{slide_id} - {mosaic_method}"
        # you can tweak bar_height, font_scale, thickness here if you like
        titled = add_title(
            img,
            title,
        )
        titled_composites.append(titled)

    # now `titled_composites` replaces `composites` below
    total = len(titled_composites)
    parts = math.ceil(total / max_per_file)
    for part in range(parts):
        start = part * max_per_file
        end   = min(start + max_per_file, total)
        chunk_imgs = titled_composites[start:end]

        if parts == 1:
            out_pdf = f"{pdf_base}.pdf"
        else:
            out_pdf = f"{pdf_base}_part{part+1}.pdf"

        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
        first, rest = chunk_imgs[0], chunk_imgs[1:]
        first.save(
            out_pdf,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=100
        )
        logging.info(f"Wrote slides {start+1}-{end} → {out_pdf}")"""

"""def generate_patch_selection_report_pdf(
    config,
    all_data,
    slide_mosaic_paths,
    mosaic_method,
    pdf_base,
    max_per_file=200
):

    # split into chunks
    all_items = list(slide_mosaic_paths.items())
    total     = len(all_items)
    if total == 0:
        logging.warning("No slides to visualize.")
        return

    parts = math.ceil(total / max_per_file)

    # choose simple vs extensive
    vizs = config.get("visualization", [])
    patch_vis = [v for v in vizs if v.startswith("patch_selection-")]

    if not patch_vis:
        logging.info("No patch_selection method specified; defaulting to 'simple'")
        chosen = "simple"
    elif len(patch_vis) > 1:
        logging.warning(f"Multiple patch_selection entries found ({patch_vis}); using first one")
        chosen = patch_vis[0].split("-", 1)[1]
    else:
        chosen = patch_vis[0].split("-", 1)[1]

    # validate
    if chosen not in PATCH_VIS_METHODS:
        logging.warning(f"Unknown patch_selection method '{chosen}'; defaulting to 'simple'")
        chosen = "simple"

    for pi in range(parts):
        start = pi * max_per_file
        end = min(start + max_per_file, total)
        chunk = all_items[start:end]

        if parts == 1:
            out_pdf = f"{pdf_base}.pdf"
        else:
            out_pdf = f"{pdf_base}_part{pi+1}.pdf"

        logging.info(f"Writing slides {start+1}-{end} → {out_pdf}")
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

        with PdfPages(out_pdf) as pdf:
            for slide_id, mosaic_pkl in chunk:
                # locate slide file
                slide_path = all_data.find_slide(slide=slide_id)
                if slide_path is None:
                    logging.warning(f"Slide not found: {slide_id}")
                    continue

                # compute dataset & roi folder from config
                ds_name   = get_dataset_name_for_slide(config, slide_path)
                roi_folder= get_path_from_dataset(config, ds_name, 'roi_path')
                mosaics_folder = os.path.dirname(mosaic_pkl)

                # call appropriate visualizer
                if chosen == "simple":
                    img = visualize_selected_patches_simple(
                        slide_id=slide_id,
                        slide_path=slide_path,
                        mosaics_folder=mosaics_folder,
                        roi_folder=roi_folder
                    )
                else:
                    img = visualize_selected_patches_extensive(
                        slide_id=slide_id,
                        slide_path=slide_path,
                        mosaics_folder=mosaics_folder,
                        roi_folder=roi_folder
                    )

                if img is None:
                    logging.warning(f"Visualization failed for {slide_id}")
                    continue

                # convert PIL image → numpy array
                arr = np.array(img)
                h, w = arr.shape[:2]
                dpi = 100

                # create a matplotlib figure just to embed into PDF
                fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
                plt.axis("off")
                plt.imshow(arr)
                plt.title(f"Slide: {slide_id}, Method: {mosaic_method}", y=1.02, fontsize=16)
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        logging.info(f"Saved {out_pdf}")"""

"""def visualize_selected_patches_simple(slide_id, slide_path, mosaics_folder, roi_folder):
    # ---- Paths ----
    mosaic_path = os.path.join(mosaics_folder, f"{slide_id}.pkl")
    roi_path    = os.path.join(roi_folder, f"{slide_id}.csv")

    # ---- Build ROI-cropped thumbnail ----
    roi_geom = load_qupath_rois(roi_path)
    minx, miny, maxx, maxy = roi_geom.bounds
    width, height = maxx - minx, maxy - miny

    slide = OpenSlide(slide_path)
    level, _ = find_best_level(slide, MAX_THUMB_SIZE)
    downsample = slide.level_downsamples[level]

    # Read the ROI at the chosen level
    lvl_w = int(width  / downsample)
    lvl_h = int(height / downsample)
    crop = slide.read_region((int(minx), int(miny)), level, (lvl_w, lvl_h)).convert("RGB")

    # Scale to fit square
    scale = min(MAX_THUMB_SIZE / lvl_w, MAX_THUMB_SIZE / lvl_h)
    tw, th = int(lvl_w * scale), int(lvl_h * scale)
    crop = crop.resize((tw, th), Image.BILINEAR)

    # Center into square canvas
    thumb_canvas = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), (255,255,255))
    ox = (MAX_THUMB_SIZE - tw) // 2
    oy = (MAX_THUMB_SIZE - th) // 2
    thumb_canvas.paste(crop, (ox, oy))

    draw = ImageDraw.Draw(thumb_canvas)
    draw.rectangle(
        [BORDER_WIDTH//2, BORDER_WIDTH//2,
        MAX_THUMB_SIZE - BORDER_WIDTH//2 - 1,
        MAX_THUMB_SIZE - BORDER_WIDTH//2 - 1],
        outline="black",
        width=BORDER_WIDTH
    )

    # ---- Load selected patches ---- 
    patch_data = load_patch_dicts_pickle(mosaic_path, reconstruct_features=False)
    tfr_path   = patch_data["properties"]["tfr_path"]
    tfr        = sf.TFRecord(tfr_path)

    patch_images = []
    for p in patch_data["patches"]:
        rec = tfr[p["tfr_index"]]
        img = sf.io.decode_image(bytes(rec["image_raw"]))
        patch_images.append(Image.fromarray(np.array(img)))

    # ---- Build composite canvas with grid of patches below the thumbnail ----
    n = len(patch_images)
    rows = math.ceil(n / GRID_COLS)
    total_height = MAX_THUMB_SIZE + rows * VIS_PATCH_SIZE

    composite = Image.new("RGB", (MAX_THUMB_SIZE, total_height), (255,255,255))
    composite.paste(thumb_canvas, (0,0))

    for i, img in enumerate(patch_images):
        r, c = divmod(i, GRID_COLS)
        composite.paste(img, (c*VIS_PATCH_SIZE, MAX_THUMB_SIZE + r*VIS_PATCH_SIZE))

    # ---- Annotate thumbnail with fixed-size squares + centered indices ----
    cv_img = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)

    # Precompute thumbnail scaling factors
    scale_x = tw / width
    scale_y = th / height

    # Draw each patch marker on the thumbnail
    for idx, p in enumerate(patch_data["patches"]):
        x_full, y_full = p["loc"]

        # Map full-resolution coordinates → thumbnail canvas coords
        dx = int((x_full - minx) * scale_x + ox)
        dy = int((y_full - miny) * scale_y + oy)

        # Draw the standardized square
        pt1 = (dx, dy)
        pt2 = (dx + MARKER_PX, dy + MARKER_PX)
        cv2.rectangle(cv_img, pt1, pt2, color=(0,0,0), thickness=-1)

        # Center the index text in that square
        text = str(idx)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        fontScale  = 1.0
        thickness  = 2
        (tw_txt, th_txt), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        text_x = dx + (MARKER_PX - tw_txt) // 2
        text_y = dy + (MARKER_PX + th_txt) // 2
        cv2.putText(
            cv_img,
            text,
            (text_x, text_y),
            font,
            fontScale,
            (255,255,255),
            thickness,
            lineType=cv2.LINE_AA
        )

    # ---- Annotate the patches grid below with their indices (optional repeat) ----
    for i in range(n):
        r, c = divmod(i, GRID_COLS)
        org = (c*VIS_PATCH_SIZE + 5, MAX_THUMB_SIZE + r*VIS_PATCH_SIZE + int(VIS_PATCH_SIZE*0.2))
        cv2.putText(
            cv_img,
            str(i),
            org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0,0,0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # 6) Convert back to PIL and display/save
    composite = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    return composite"""

"""def get_path_from_dataset(config: Dict, dataset_name: str, path: str) -> str:
    for ds in config.get("datasets", []):
        if ds.get("name") == dataset_name:
            roi = ds.get(path)
            if roi:
                return os.path.abspath(roi)
            else:
                raise KeyError(f"Dataset '{dataset_name}' has no roi_path defined.")
    raise KeyError(f"No dataset named '{dataset_name}' in config.")"""


"""def visualize_selected_patches_extensive(slide_id, slide_path, mosaics_folder, roi_folder, patch_px, patch_um):

    # ---- Paths ----
    mosaic_path = os.path.join(mosaics_folder, f"{slide_id}.pkl")
    mosaic_patch_ids_path = os.path.join(mosaics_folder, f"{slide_id}.npz")
    roi_path = os.path.join(roi_folder,    f"{slide_id}.csv")

    # ---- Build ROI‐cropped thumbnail ---- 
    roi_geom = load_qupath_rois(roi_path)
    minx, miny, maxx, maxy = roi_geom.bounds
    width, height = maxx - minx, maxy - miny

    slide = OpenSlide(slide_path)
    level, _ = find_best_level(slide, MAX_THUMB_SIZE)
    downsample = slide.level_downsamples[level]

    slide_mpp = float(
        slide.properties.get("openslide.mpp-x",
        slide.properties.get("tiff_mpp_x", 1.0))
    )

    # --- 3) Compute full-res pixel size of each patch box ---
    extraction_mpp   = patch_um / patch_px            # µm per pixel at extraction
    scale_factor     = extraction_mpp / slide_mpp     # how extraction‐MPP compares to slide‐MPP
    fullres_patch_px = int(round(patch_px * scale_factor))

    lvl_w = int(width  / downsample)
    lvl_h = int(height / downsample)
    crop = slide.read_region((int(minx), int(miny)), level, (lvl_w, lvl_h)).convert("RGB")

    scale = min(MAX_THUMB_SIZE / lvl_w, MAX_THUMB_SIZE / lvl_h)
    tw, th = int(lvl_w * scale), int(lvl_h * scale)
    crop = crop.resize((tw, th), Image.BILINEAR)

    base_thumb = Image.new("RGB", (MAX_THUMB_SIZE, MAX_THUMB_SIZE), (255,255,255))
    ox = (MAX_THUMB_SIZE - tw)//2
    oy = (MAX_THUMB_SIZE - th)//2
    base_thumb.paste(crop, (ox, oy))
    base_cv = cv2.cvtColor(np.array(base_thumb), cv2.COLOR_RGB2BGR)

    # Precompute scaling
    scale_x = tw / width
    scale_y = th / height

    # Prepare two annotated thumbnails
    thumb_bin = base_cv.copy()
    thumb_idx = base_cv.copy()

    # ---- Annotate thumb_bin with colored bin overlays + legend ----

    # Load bin data
    bin_data     = np.load(mosaic_patch_ids_path)
    bin_ids      = bin_data['bin_ids']
    coords       = bin_data['coords']
    merged_polys = merge_patch_polygons_by_bin(coords, bin_ids, fullres_patch_px)

    overlay     = thumb_bin.copy()
    unique_bins = sorted(merged_polys.keys())

    unique_bins = sorted(merged_polys.keys())
    palette = generate_distinct_bgr_colors(len(unique_bins))

    for i, bin_id in enumerate(unique_bins):
        color = palette[i % len(palette)]
        poly  = merged_polys[bin_id]
        geoms = [poly] if isinstance(poly, Polygon) else poly.geoms
        for geom in geoms:
            pts = [
                [int((x - minx)*scale_x + ox), int((y - miny)*scale_y + oy)]
                for x,y in geom.exterior.coords
            ]
            pts = np.array(pts, np.int32).reshape((-1,1,2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, color, 2)
    thumb_bin = cv2.addWeighted(overlay, PATCH_ALPHA, thumb_bin, 1-PATCH_ALPHA, 0)

    # legend inside thumb_bin
    # compute text width for safety
    max_text_width = max(
        cv2.getTextSize(f"bin {b}", FONT, 1.0, 2)[0][0] 
        for b in unique_bins
    )

    # right-side, vertically centered
    legend_x = MAX_THUMB_SIZE - LEGEND_MARGING - SWATCH_SIZE - max_text_width - 10
    total_legend_height = len(unique_bins)*SWATCH_SIZE + (len(unique_bins)-1)*SWATCH_PAD
    legend_y = (MAX_THUMB_SIZE - total_legend_height) // 2

    for i, bin_id in enumerate(unique_bins):
        color = palette[i % len(palette)]
        y0    = legend_y + i*(SWATCH_SIZE + SWATCH_PAD)
        # swatch
        cv2.rectangle(
            thumb_bin,
            (legend_x, y0),
            (legend_x + SWATCH_SIZE, y0 + SWATCH_PAD),
            color, -1
        )
        # label
        cv2.putText(
            thumb_bin,
            f"bin {bin_id}",
            (legend_x + SWATCH_SIZE + 5, y0 + SWATCH_SIZE - 5),
            FONT, 1.0, (0,0,0), 2, cv2.LINE_AA
        )

    # ---- Annotate thumb_idx with patch markers + centered indices ----
    patch_data = load_patch_dicts_pickle(mosaic_path, reconstruct_features=False)
    for idx, p in enumerate(patch_data["patches"]):
        x_full, y_full = p["loc"]
        dx = int((x_full - minx)*scale_x + ox)
        dy = int((y_full - miny)*scale_y + oy)
        cv2.rectangle(thumb_idx, (dx, dy), (dx+MARKER_PX, dy+MARKER_PX), (0,0,0), -1)
        txt = str(idx)
        (w,h),_ = cv2.getTextSize(txt, FONT, 1.0, 2)
        tx = dx + (MARKER_PX-w)//2
        ty = dy + (MARKER_PX+h)//2
        cv2.putText(thumb_idx, txt, (tx, ty), FONT, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # ---- Load patch images ----
    patch_data   = load_patch_dicts_pickle(mosaic_path, reconstruct_features=False)
    tfr_path     = patch_data["properties"]["tfr_path"]
    tfr          = sf.TFRecord(tfr_path)
    patch_images = []
    for p in patch_data["patches"]:
        rec = tfr[p["tfr_index"]]
        img = sf.io.decode_image(bytes(rec["image_raw"]))
        patch_images.append(Image.fromarray(np.array(img)))

    # ---- Composite: two thumbs side-by-side + full-width grid ----
    patch_grid_cols = GRID_COLS * 2
    n               = len(patch_images)
    rows            = math.ceil(n / patch_grid_cols)
    total_h         = MAX_THUMB_SIZE + rows * VIS_PATCH_SIZE
    composite       = Image.new("RGB", (2*MAX_THUMB_SIZE, total_h), (0,0,0))

    # ---- Add borders ----
    cv2.rectangle(
        thumb_bin,
        (0, 0),
        (MAX_THUMB_SIZE-1, MAX_THUMB_SIZE-1),
        color=(0, 0, 0),
        thickness=BORDER_WIDTH
    )
    cv2.rectangle(
        thumb_idx,
        (0, 0),
        (MAX_THUMB_SIZE-1, MAX_THUMB_SIZE-1),
        color=(0, 0, 0),
        thickness=BORDER_WIDTH
    )

    # ---- paste thumbs ----
    composite.paste(Image.fromarray(cv2.cvtColor(thumb_bin, cv2.COLOR_BGR2RGB)), (0,0))
    composite.paste(Image.fromarray(cv2.cvtColor(thumb_idx, cv2.COLOR_BGR2RGB)),
                    (MAX_THUMB_SIZE,0))

    # paste patch grid
    for i, img in enumerate(patch_images):
        r, c = divmod(i, patch_grid_cols)
        x    = c * VIS_PATCH_SIZE
        y    = MAX_THUMB_SIZE + r * VIS_PATCH_SIZE
        composite.paste(img, (x, y))

    # ---- Annotate grid patches with their indices ----
    cv_img = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
    for i in range(n):
        r, c = divmod(i, patch_grid_cols)
        x = c * VIS_PATCH_SIZE
        y = MAX_THUMB_SIZE + r * VIS_PATCH_SIZE
        org = (x + 5, y + int(VIS_PATCH_SIZE*0.15))
        cv2.putText(cv_img, str(i), org, FONT, 1.0, (0,0,0), 2, cv2.LINE_AA)

    # ---- Convert back to PIL and display ----
    composite = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    return composite"""