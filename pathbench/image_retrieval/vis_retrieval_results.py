import math
import os

import cv2
import numpy as np
from PIL import Image
import openslide
from typing import Optional, Dict
import json
from functools import lru_cache
import re
import unicodedata

from .vis_utils import crop_roi, get_dataset_name_for_slide, get_path_from_dataset, load_qupath_rois, find_best_level

# ─────────────────────────── Constants ────────────────────────────────────
GRID_COLS   = 5
GRID_ROWS   = 2        # always show up to 10 hits
THUMB_SIZE  = 512      # image cell interior
TEXT_H      = 40       # text band under query (unused for hits)
HIT_TEXT_H  = 512       # text band under hits
CELL_W      = THUMB_SIZE
CELL_H_HIT  = THUMB_SIZE + HIT_TEXT_H
BORDER_PX   = 2

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.7
THICKNESS   = 1
TEXT_COLOR  = (0,0,0)
BG_COLOR    = (255,255,255)

QUERY_TEXT_W = 512     # width of the panel next to query image

ADD_METADATA = ['subtype',
                'Diagnostic IHC',
                'ISH type',
                'Diagnostic ISH',
                'Molecular Panel',
                'Diagnostic Molecular']

# page dimensions
PAGE_W      = CELL_W * GRID_COLS
PAGE_H      = THUMB_SIZE + CELL_H_HIT * GRID_ROWS
# ──────────────────────────────────────────────────────────────────────────

def _load_roi_crop(slide_path, roi_csv, target_w, target_h):
    """ROI-crop, choose best level so that slide→thumbnail ≤ target_w×target_h, then scale."""
    slide = openslide.OpenSlide(slide_path)
    try:
        geom = load_qupath_rois(roi_csv)
        minx, miny, maxx, maxy = geom.bounds
        W, H = maxx-minx, maxy-miny

        lvl, _ = find_best_level(slide, max(target_w, target_h))
        ds     = slide.level_downsamples[lvl]
        lvl_w  = int(W/ds); lvl_h = int(H/ds)

        crop   = slide.read_region((int(minx),int(miny)), lvl, (lvl_w, lvl_h)).convert("RGB")
    finally:
        slide.close()

    scale = min(target_w/lvl_w, target_h/lvl_h)
    tw, th = int(lvl_w*scale), int(lvl_h*scale)
    return crop.resize((tw,th), Image.BILINEAR), (W,H), (tw,th)

def _to_ascii(s: str) -> str:
    # strip accents and drop unsupported glyphs
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

@lru_cache(maxsize=64)
def _load_metadata_file(metadata_path: str):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_metadata_lines(config, slide_id, slide_path, slide_label):

    ds_name = get_dataset_name_for_slide(config, slide_path)
    metadata_path = get_path_from_dataset(config, ds_name, "metadata_path")
    metadata = _load_metadata_file(metadata_path)

    lines = [f"Slide: {slide_id}", f"Label: {slide_label}"]
    slide_metadata = metadata.get(slide_id, {})

    for key in ADD_METADATA:
        value = slide_metadata.get(key, '-')
        meta_name = key[:1].upper() + key[1:] if key else key
        lines.append(f"{meta_name}: {value}")

    return lines

def _text_size(text: str, font, scale: float, thickness: int):
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w, h

def _wrap_line_by_width(text: str, max_w: int, font, scale: float, thickness: int):
    """
    Wrap a single line to multiple lines so each fits within max_w pixels.
    Preserves spaces between words; if a 'word' itself exceeds max_w,
    it gets split into character chunks.
    """
    if not text:
        return [""]

    # Split into tokens that include spaces as tokens
    tokens = re.split(r'(\s+)', text)

    lines = []
    cur = ""
    for tok in tokens:
        if tok == "":
            continue
        # Try adding token to current line
        candidate = cur + tok
        w, _ = _text_size(candidate, font, scale, thickness)
        if w <= max_w:
            cur = candidate
        else:
            # If current is empty, the token itself is too wide -> hard wrap token
            if cur.strip() == "":
                # Split token into chunks that fit
                chunk = ""
                for ch in tok:
                    cw, _ = _text_size(chunk + ch, font, scale, thickness)
                    if cw <= max_w:
                        chunk += ch
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                if chunk:
                    cur = chunk  # start next line with remainder
                else:
                    cur = ""
            else:
                # Push current line, start new with tok (or its wrapped version if still too wide)
                lines.append(cur.rstrip())
                cur = ""
                # Now place tok onto a new line (may still be too wide)
                w_tok, _ = _text_size(tok, font, scale, thickness)
                if w_tok <= max_w:
                    cur = tok.lstrip()
                else:
                    # hard-wrap tok as above
                    chunk = ""
                    for ch in tok:
                        cw, _ = _text_size(chunk + ch, font, scale, thickness)
                        if cw <= max_w:
                            chunk += ch
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    cur = chunk

    if cur.strip() != "" or (cur != "" and not lines):
        lines.append(cur.rstrip())
    return lines

def _wrap_lines(lines, max_w: int, font, scale: float, thickness: int):
    wrapped = []
    for ln in lines:
        wrapped.extend(_wrap_line_by_width(ln, max_w, font, scale, thickness))
    return wrapped

def visualize_retrieval_result(
    result: dict,
    config: dict,
    all_data,
) -> Image.Image:
    """
    Build a PAGE_W×dynamic_H composite: centered query (512×512) + 512px wrapped text panel,
    with both query and hit metadata constrained in height. When text exceeds available
    vertical space, it is truncated and an ellipsis ("…") is appended as the last line.
    """

    # 1) unpack
    qid    = result["query_slide_id"]
    qlab   = result["query_label"]
    hits   = result.get("top_k", [])[: GRID_COLS * GRID_ROWS]

    # 2) start blank canvas for query (fixed height: THUMB_SIZE)
    canvas_query = np.full((THUMB_SIZE, PAGE_W, 3), BG_COLOR, np.uint8)

    # 3) compute query‐panel origin
    panel_w = THUMB_SIZE + QUERY_TEXT_W
    x0      = (PAGE_W - panel_w) // 2
    y0      = 0

    # 4) draw borders for query image & text panel
    cv2.rectangle(canvas_query,
                  (x0, y0),
                  (x0+THUMB_SIZE-1, y0+THUMB_SIZE-1),
                  TEXT_COLOR, BORDER_PX)
    cv2.rectangle(canvas_query,
                  (x0+THUMB_SIZE, y0),
                  (x0+panel_w-1, y0+THUMB_SIZE-1),
                  TEXT_COLOR, BORDER_PX)

    # 5) crop & paste query thumbnail
    qpath = all_data.find_slide(slide=qid)
    qimg, (Wq,Hq), _ = crop_roi(config, qpath, qid, THUMB_SIZE, BORDER_PX)
    qp = np.array(qimg)[:,:,::-1]  # BGR

    # center inside THUMB_SIZE
    oy_q = (THUMB_SIZE - qp.shape[0])//2
    ox_q = (THUMB_SIZE - qp.shape[1])//2
    canvas_query[
        y0+oy_q : y0+oy_q+qp.shape[0],
        x0+ox_q : x0+ox_q+qp.shape[1]
    ] = qp

    # 6) draw metadata text in the right panel (wrap width + cap height with ellipsis)
    qlines = get_metadata_lines(config, qid, qpath, qlab)
    (_, txt_h), _ = cv2.getTextSize("Ay", FONT, FONT_SCALE, THICKNESS)
    line_sp = 5

    # Width constraint for query text panel
    tx0 = x0 + THUMB_SIZE + 10
    max_text_w_query = QUERY_TEXT_W - 20  # left/right padding inside query panel
    wrapped_q = _wrap_lines(qlines, max_text_w_query, FONT, FONT_SCALE, THICKNESS)

    # Height constraint for query panel
    top_pad = 20
    bottom_pad = 10
    usable_h_query = THUMB_SIZE - top_pad - bottom_pad
    lines_fit_query = max(1, usable_h_query // (txt_h + line_sp))
    if len(wrapped_q) > lines_fit_query:
        wrapped_q = wrapped_q[:max(1, lines_fit_query - 1)] + ["…"]

    # Render query text
    ty = y0 + top_pad + txt_h
    for ln in wrapped_q:
        ln = _to_ascii(ln).replace("…", "...")
        cv2.putText(canvas_query, ln, (tx0, ty),
                    FONT, FONT_SCALE, TEXT_COLOR, THICKNESS,
                    lineType=cv2.LINE_AA)
        ty += txt_h + line_sp

    # --- 7) Hits grid with fixed band height + ellipsis when needed ---
    (_, txt_h), _ = cv2.getTextSize("Ay", FONT, FONT_SCALE, THICKNESS)
    line_sp = 5
    text_left_pad = 5
    max_text_w_hit = THUMB_SIZE - 2*text_left_pad  # keep some horizontal padding

    # Use the existing constant to cap band height
    MAX_HIT_BAND_H = HIT_TEXT_H  # e.g., 80 px

    # Wrap and cap lines for each hit; store computed band heights (fixed) and wrapped lines
    hit_lines_wrapped, hit_band_hs = [], []
    for hit in hits:
        hid   = hit["slide_id"]
        hlab  = hit["label"]
        hpath = all_data.find_slide(slide=hid)
        lines = get_metadata_lines(config, hid, hpath, hlab)

        wrapped = _wrap_lines(lines, max_text_w_hit, FONT, FONT_SCALE, THICKNESS)

        # Height constraint for hit band
        usable_h_hit = MAX_HIT_BAND_H - line_sp  # account for top spacing
        lines_fit_hit = max(1, usable_h_hit // (txt_h + line_sp))
        if len(wrapped) > lines_fit_hit:
            wrapped = wrapped[:max(1, lines_fit_hit - 1)] + ["…"]

        hit_lines_wrapped.append(wrapped)
        hit_band_hs.append(MAX_HIT_BAND_H)  # fixed height per design

    # 7b) per-row max band (all equal to MAX_HIT_BAND_H now; still robust for partial last row)
    n_hits = len(hits)
    n_rows = math.ceil(n_hits/GRID_COLS) if n_hits > 0 else 0
    row_max = []
    for r in range(n_rows):
        start = r*GRID_COLS
        end   = min(start+GRID_COLS, n_hits)
        row_max.append(max(hit_band_hs[start:end] or [0]))

    # 7c) compute full canvas height (top block + rows of hits with their bands)
    total_h = THUMB_SIZE + sum(THUMB_SIZE + h for h in row_max)

    # 7d) build new canvas and copy query row
    canvas = np.full((total_h, PAGE_W, 3), BG_COLOR, np.uint8)
    canvas[0:THUMB_SIZE, :, :] = canvas_query

    # compute Y offsets for rows of hits
    row_y = [THUMB_SIZE]
    for r in range(1, n_rows):
        prev = row_y[r-1] + THUMB_SIZE + row_max[r-1]
        row_y.append(prev)

    # 7e) draw each hit (thumbnail + fixed-height text band)
    for i, hit in enumerate(hits):
        r, c   = divmod(i, GRID_COLS)
        xh     = c * CELL_W
        yh     = row_y[r]
        wrapped = hit_lines_wrapped[i]
        band_h  = hit_band_hs[i]

        # border around image
        cv2.rectangle(canvas,
                      (xh, yh),
                      (xh+THUMB_SIZE-1, yh+THUMB_SIZE-1),
                      TEXT_COLOR, BORDER_PX)

        # crop & paste hit ROI
        hid   = hit["slide_id"]
        hpath = all_data.find_slide(slide=hid)
        himg, _, _ = crop_roi(config, hpath, hid, THUMB_SIZE, BORDER_PX)
        hip = np.array(himg)[:,:,::-1]

        oy_h = (THUMB_SIZE - hip.shape[0])//2
        ox_h = (THUMB_SIZE - hip.shape[1])//2
        canvas[
            yh+oy_h : yh+oy_h+hip.shape[0],
            xh+ox_h : xh+ox_h+hip.shape[1]
        ] = hip

        # border around text band (fixed height)
        band_y0 = yh + THUMB_SIZE
        band_y1 = band_y0 + band_h
        cv2.rectangle(canvas,
                      (xh, band_y0),
                      (xh+THUMB_SIZE-1, band_y1-1),
                      TEXT_COLOR, BORDER_PX)

        # render wrapped (and possibly truncated) text with ellipsis
        ty2 = band_y0 + line_sp + txt_h
        for ln in wrapped:
            ln = _to_ascii(ln).replace("…", "...")
            cv2.putText(canvas, ln, (xh + text_left_pad, ty2),
                        FONT, FONT_SCALE, TEXT_COLOR, THICKNESS,
                        lineType=cv2.LINE_AA)
            ty2 += txt_h + line_sp

    # 8) back to PIL
    return Image.fromarray(canvas[:,:,::-1])

def generate_image_retrieval_report_pdf(
    config: dict,
    results: list,
    all_data,
    output_dir: str,
    base_name: str = "retrieval_report",
    per_file_limit: int = 200
):
    """
    Group by true class, generate ROI-cropped PIL pages & save as multi-page PDFs.
    """
    os.makedirs(output_dir, exist_ok=True)
    by_cls = {}
    for r in results:
        lbl = r.get("query_label","UNKNOWN")
        by_cls.setdefault(lbl,[]).append(r)

    for cls, slides in by_cls.items():
        safe = cls.replace(" ","_")
        parts= math.ceil(len(slides)/per_file_limit)
        for pi in range(parts):
            chunk = slides[pi*per_file_limit:(pi+1)*per_file_limit]
            if parts==1:
                out = os.path.join(output_dir, f"{base_name}_{safe}.pdf")
            else:
                out = os.path.join(output_dir, f"{base_name}_{safe}_part{pi+1}.pdf")

            pages = []
            for res in chunk:
                im = visualize_retrieval_result(res, config, all_data)
                if im:
                    pages.append(im)

            if pages:
                first, rest = pages[0].convert("RGB"), [p.convert("RGB") for p in pages[1:]]
                first.save(out, "PDF", save_all=True, append_images=rest, resolution=100)