import openslide
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

from ..image_retrieval.utils import load_patch_dicts_pickle

def visualize_patch_selection_overview(
    pdf,
    all_data,
    slide_ids: list,
    slide_mosaic_paths: dict,
    thumb_size: int = 512,
    tile_size: int = 256,
    dpi: int = 80,
    inflation: float = 1.2,
    linewidth: float = 3.0
) -> None:
    """
    Add overview pages to `pdf`: each page shows up to 3 slide thumbnails
    (stacked vertically) with inflated red squares marking selected patches.

    Args:
        pdf:               an open PdfPages to append to.
        all_data:          slideflow Dataset for locating slide files.
        slide_ids:         list of slide IDs in display order.
        slide_mosaic_paths:map slide_id → mosaic .pkl.
        thumb_size:        max thumbnail dimension in px.
        tile_size:         patch width/height in px.
        dpi:               DPI for pdf.savefig.
        inflation:         factor to enlarge each patch square.
        linewidth:         thickness of the rectangle edges.
    """
    slides_per_page = 3
    for start in range(0, len(slide_ids), slides_per_page):
        chunk = slide_ids[start:start + slides_per_page]
        rows = len(chunk)

        # One column, rows rows
        fig, axes = plt.subplots(rows, 1, figsize=(8.5, 11))
        if rows == 1:
            axes = [axes]

        for ax, slide_id in zip(axes, chunk):
            # load slide + thumbnail
            path = all_data.find_slide(slide=slide_id)
            if path is None:
                ax.axis("off")
                continue
            slide = openslide.OpenSlide(path)
            thumb = slide.get_thumbnail((thumb_size, thumb_size))
            W, H = slide.dimensions
            sx, sy = thumb.width / W, thumb.height / H

            # draw thumbnail
            ax.imshow(thumb)
            ax.set_title(slide_id, fontsize=10)
            ax.axis("off")

            # load mosaic
            mosaic = load_patch_dicts_pickle(slide_mosaic_paths[slide_id])
            for p in mosaic["patches"]:
                x, y = p["loc"]
                w = tile_size * sx
                h = tile_size * sy
                # inflate
                dx = (inflation - 1) * w / 2
                dy = (inflation - 1) * h / 2

                rect = Rectangle(
                    (x * sx - dx, y * sy - dy),
                    w * inflation, h * inflation,
                    linewidth=linewidth,
                    edgecolor="red",
                    facecolor="none"
                )
                ax.add_patch(rect)

        # rasterize images to keep PDF small
        for a in fig.axes:
            for im in a.get_images():
                im.set_rasterized(True)

        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def visualize_patch_selection_details(
    pdf: PdfPages,
    slide_ids: list,
    slide_mosaic_paths: dict,
    num_patches: int = 10,
    patch_size: int = 256
) -> None:
    """
    Add one detail page per slide: a single row of up to `num_patches` randomly sampled patch images.

    Args:
        pdf: open PdfPages to append to.
        slide_ids: list of slide IDs.
        slide_mosaic_paths: map slide_id → mosaic .pkl.
        num_patches: max crops to show per slide.
        patch_size: not used for layout, but kept for signature consistency.
    """
    for slide_id in slide_ids:
        mosaic_pkl = slide_mosaic_paths[slide_id]
        try:
            mosaic = load_patch_dicts_pickle(mosaic_pkl)
        except Exception as e:
            logging.warning(f"Cannot load mosaic {slide_id}: {e}")
            continue

        patches = mosaic["patches"]
        if not patches:
            logging.warning(f"No patches for {slide_id}")
            continue

        chosen = random.sample(patches, min(num_patches, len(patches)))
        tfr = sf.TFRecord(mosaic["properties"]["tfr_path"])

        cols = len(chosen)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 1.5, 2))
        if cols == 1:
            axes = [axes]

        # remove all spacing between subplots
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0, hspace=0)

        for ax, p in zip(axes, chosen):
            rec = tfr[p["tfr_index"]]
            img = sf.io.decode_image(bytes(rec["image_raw"]))
            ax.imshow(np.array(img))
            ax.axis("off")

        fig.suptitle(slide_id, fontsize=10)
        plt.tight_layout(pad=1)

        for sub in fig.axes:
            for im in sub.get_images():
                im.set_rasterized(True)

        pdf.savefig(fig, dpi=100, bbox_inches="tight")
        plt.close(fig)

def generate_simple_patch_selection_report_pdf(
    config: dict,
    all_data,
    slide_mosaic_paths: dict,
    pdf_path: str,
    thumb_size: int = 512,
    num_detail_patches: int = 10,
    tile_size: int = None
) -> None:
    """
    Create a single PDF where each page contains three blocks of:
      • an overview thumbnail with red patch rectangles (5% oversized)
      • a row of up to `num_detail_patches` example crops, touching each other

    Args:
        config (dict): Experiment configuration (for tile_px if tile_size None).
        all_data: slideflow Dataset for locating slides.
        slide_mosaic_paths (dict): slide_id → mosaic.pkl path.
        pdf_path (str): output PDF path.
        thumb_size (int): max side length for thumbnails.
        num_detail_patches (int): number of random patches per block.
        tile_size (int, optional): patch width/height in px.
    """
    # derive tile_size if needed
    if tile_size is None:
        tpx = config["benchmark_parameters"]["tile_px"]
        tile_size = tpx[0] if isinstance(tpx, (list,tuple)) else tpx

    slide_ids = list(slide_mosaic_paths.keys())
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(slide_ids), 3):
            chunk = slide_ids[i:i+3]
            fig = plt.figure(figsize=(11, 8.5))
            # 6 rows: overview, detail, overview, detail, ...
            gs = GridSpec(6, num_detail_patches,
                          figure=fig,
                          height_ratios=[3, 1] * len(chunk),
                          wspace=0, hspace=0.5)
            # expand to full page
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

            for bi, slide_id in enumerate(chunk):
                # overview
                ax_thumb = fig.add_subplot(gs[2*bi, :])
                slide_path = all_data.find_slide(slide=slide_id)
                if slide_path:
                    slide = openslide.OpenSlide(slide_path)
                    thumb = slide.get_thumbnail((thumb_size, thumb_size))
                    W, H = slide.dimensions
                    sx, sy = thumb.width / W, thumb.height / H

                    ax_thumb.imshow(thumb)
                    ax_thumb.set_title(slide_id, fontsize=10)
                    ax_thumb.axis("off")

                    mosaic = load_patch_dicts_pickle(slide_mosaic_paths[slide_id])
                    pad_x = 0.05 * tile_size * sx
                    pad_y = 0.05 * tile_size * sy

                    for p in mosaic["patches"]:
                        x, y = p["loc"]
                        rect = Rectangle(
                            (x * sx - pad_x, y * sy - pad_y),
                            tile_size * sx + 2*pad_x,
                            tile_size * sy + 2*pad_y,
                            edgecolor="red", facecolor="none", linewidth=1.5
                        )
                        ax_thumb.add_patch(rect)
                else:
                    ax_thumb.text(0.5,0.5,"Slide not found",ha="center",va="center")
                    ax_thumb.axis("off")

                # detail row
                mosaic = load_patch_dicts_pickle(slide_mosaic_paths[slide_id])
                patches = mosaic["patches"]
                chosen = random.sample(patches, min(num_detail_patches, len(patches)))
                tfr = sf.TFRecord(mosaic["properties"]["tfr_path"])

                for j in range(num_detail_patches):
                    ax = fig.add_subplot(gs[2*bi + 1, j])
                    if j < len(chosen):
                        rec = tfr[chosen[j]["tfr_index"]]
                        img = sf.io.decode_image(bytes(rec["image_raw"]))
                        ax.imshow(np.array(img))
                    ax.axis("off")

            # rasterize to shrink size
            for ax in fig.axes:
                for im in ax.get_images():
                    im.set_rasterized(True)

            pdf.savefig(fig, dpi=100, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved combined PDF report to: {pdf_path}")

##########################################################################################################################################################
##########################################################################################################################################################

def visualize_selected_patches_slide(
    slide_id,
    slide_path,
    mosaic_data,
    method,
    patch_size=256,
    thumb_size=800,       # ↓ lowered from 1024 to 800 to shrink PDF
    save_path=None
):
    """
    Draw a single figure showing:
      1) the full‐slide thumbnail with red boxes around each selected patch, and
      2) each selected patch crop underneath, labeled by index.
    All image axes (thumbnail and patch crops) are forced to rasterize,
    so that the final PDF embeds them as bitmaps at modest dpi.
    """
    try:
        # ---- unpack the mosaic_data ----
        mosaic_patches    = mosaic_data["patches"]
        mosaic_properties = mosaic_data["properties"]

        # ---- open TFRecord to decode patch images later ----
        tfr = sf.TFRecord(mosaic_properties["tfr_path"])

        # ---- load the WSI and grab a downscaled thumbnail ----
        slide = openslide.OpenSlide(slide_path)
        thumb = slide.get_thumbnail((thumb_size, thumb_size))
        dims = slide.dimensions
        scale_x = thumb.width / dims[0]
        scale_y = thumb.height / dims[1]

        num_patches = len(mosaic_patches)
        num_cols = min(6, num_patches)
        num_rows = math.ceil(num_patches / num_cols)

        # ---- build a GridSpec: first row=thumbnail, then a small spacer, then patches ----
        fig = plt.figure(figsize=(12, 3 + 1.5 * num_rows))
        gs = GridSpec(2 + num_rows, num_cols, figure=fig,
                      height_ratios=[4, 0.1] + [1]*num_rows)

        # ---- Plot the thumbnail ----
        ax_thumb = fig.add_subplot(gs[0, :])
        ax_thumb.imshow(thumb)
        ax_thumb.set_title(f"{slide_id} – Selected patches ({method})", fontsize=10)
        ax_thumb.axis("off")

        # ---- draw red rectangles + labels on top of the thumbnail ----
        for i, patch in enumerate(mosaic_patches):
            x, y = patch['loc']
            rect = Rectangle(
                (x * scale_x, y * scale_y),
                patch_size * scale_x, patch_size * scale_y,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax_thumb.add_patch(rect)
            ax_thumb.text(
                x * scale_x, y * scale_y,
                str(i + 1), color='white', fontsize=6,
                backgroundcolor='red', va='top'
            )

        # ↓↓↓ Force entire thumbnail + vector overlays to become one bitmap ↓↓↓
        ax_thumb.set_rasterized(True)

        # ---- Plot each patch crop below the thumbnail ----
        for i, patch in enumerate(mosaic_patches):
            row = 2 + (i // num_cols)
            col = i % num_cols
            ax = fig.add_subplot(gs[row, col])

            record = tfr[patch['tfr_index']]
            img = sf.io.decode_image(bytes(record['image_raw']))
            ax.imshow(np.array(img))
            ax.axis("off")
            ax.set_title(f"{i + 1}", fontsize=6)

            # ↓↓↓ Force each patch crop to be rasterized ↓↓↓
            # (Matplotlib’s imshow already draws as an "AxesImage", but this
            #  ensures any future overlays are flattened.)
            ax.set_rasterized(True)

        plt.tight_layout()

        # ---- Save to disk or return the figure ----
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Use dpi=80 or 100—anything higher just bloats the file size
            fig.savefig(save_path, dpi=80, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Saved patch visualization to: {save_path}")
            return None

        return fig

    except Exception as e:
        logging.warning(f"Visualization failed for {slide_id}: {e}")
        return None

def generate_extensive_patch_selection_report_pdf(
    config,
    all_data,
    slide_mosaic_paths,
    mosaic_method,
    pdf_base,
    patch_size=None,
    max_per_file=200
):
    """
    Iterate over every slide_id → mosaic pickle, load both, generate the figure
    with `visualize_selected_patches_slide`, and write each page into a PDF.
    Splits into multiple PDF files if there are more than `max_per_file` slides.

    Args:
        config (dict):
            Experiment configuration dictionary.
        all_data (sf.Dataset):
            Slideflow dataset object providing TFRecord file paths.
        slide_mosaic_paths (dict[str,str]):
            Mapping slide_id → path to mosaic‐pickle.
        mosaic_method (str):
            Method name used for labeling in the titles.
        pdf_path (str):
            Base path (no extension) where output PDF(s) will be saved.
            If more than `max_per_file` slides, files will be named
            `{pdf_path}_part1.pdf`, `{pdf_path}_part2.pdf`, etc.
        patch_size (int, optional):
            Side length of each patch in pixels (defaults to first tile_px).
        max_per_file (int):
            Maximum number of slides (i.e. pages) per PDF. Default = 200.
    """
    # If patch_size wasn’t given, infer from config
    if patch_size is None:
        px_list = config['benchmark_parameters']['tile_px']
        patch_size = px_list[0] if isinstance(px_list, (list, tuple)) else px_list

    # Gather all slide IDs in a list so we can slice into chunks of 200
    all_items = list(slide_mosaic_paths.items())
    total_slides = len(all_items)
    if total_slides == 0:
        logging.warning("No slides found in slide_mosaic_paths → nothing to visualize.")
        return

    # Compute how many “parts” (PDFs) we need
    num_parts = math.ceil(total_slides / max_per_file)

    for part_idx in range(num_parts):
        # Determine slice boundaries:
        start = part_idx * max_per_file
        end = min(start + max_per_file, total_slides)
        chunk = all_items[start:end]

        # Build a filename for this chunk:
        if num_parts == 1:
            out_pdf = f"{pdf_base}.pdf"
        else:
            out_pdf = f"{pdf_base}_part{part_idx+1}.pdf"

        logging.debug(f"Writing slides {start}–{end-1} (of {total_slides}) to {out_pdf}")

        # Open a PdfPages for this chunk
        with PdfPages(out_pdf) as pdf:
            for slide_id, mosaic_pkl in chunk:
                # Attempt to find the slide’s filesystem path:
                slide_path = all_data.find_slide(slide=slide_id)
                if slide_path is None:
                    logging.warning(f"Slide path not found for slide ID: {slide_id}")
                    continue

                # Load the “mosaic” dictionary (patch indices + TFRecord path)
                try:
                    mosaic_data = load_patch_dicts_pickle(mosaic_pkl)
                except Exception as e:
                    logging.warning(f"Failed to load mosaic for {slide_id}: {e}")
                    continue

                # Produce the figure (thumbnail + patch grid) via your existing helper
                fig = visualize_selected_patches_slide(
                    slide_id=slide_id,
                    slide_path=slide_path,
                    mosaic_data=mosaic_data,
                    method=mosaic_method,
                    patch_size=patch_size
                )
                if fig is None:
                    continue

                # Rasterize every image in the figure so the PDF stays small
                for ax in fig.axes:
                    for im in ax.get_images():
                        im.set_rasterized(True)

                # Save this page to the current PDF at 80 dpi
                pdf.savefig(fig, dpi=80, bbox_inches='tight')
                plt.close(fig)

        logging.debug(f"Completed writing {out_pdf}")

    logging.debug(f"All done. Created {num_parts} PDF file(s).")

"""def generate_extensive_patch_selection_report_pdf(
    config,
    all_data,
    slide_mosaic_paths,
    mosaic_method,
    pdf_path,
    patch_size=None
):
    
    Iterate over every slide_id → mosaic pickle, load both, generate the figure
    with `visualize_selected_patches_slide`, and write each page into a single PDF.
    By rasterizing each axes and saving at a low DPI, the final PDF is much smaller.

    # Derive patch_size from config if not given
    if patch_size is None:
        px_list = config['benchmark_parameters']['tile_px']
        patch_size = px_list[0] if isinstance(px_list, (list, tuple)) else px_list

    with PdfPages(pdf_path) as pdf:
        for slide_id, mosaic_pkl in slide_mosaic_paths.items():
            slide_path = all_data.find_slide(slide=slide_id)
            if slide_path is None:
                logging.warning(f"Slide path not found for slide ID: {slide_id}")
                continue

            # Load the “mosaic” dictionary (patch indices + TFRecord path)
            try:
                mosaic_data = load_patch_dicts_pickle(mosaic_pkl)
            except Exception as e:
                logging.warning(f"Failed to load mosaic for {slide_id}: {e}")
                continue

            fig = visualize_selected_patches_slide(
                slide_id=slide_id,
                slide_path=slide_path,
                mosaic_data=mosaic_data,
                method=mosaic_method,
                patch_size=patch_size
            )
            if fig is None:
                continue

            # Make sure **all** image artists get rasterized (thumbnail & patches)
            for ax in fig.axes:
                for im in ax.get_images():
                    im.set_rasterized(True)

            pdf.savefig(fig, dpi=80, bbox_inches='tight')
            plt.close(fig)

    logging.info(f"Saved multi‐page PDF of patch visualizations to: {pdf_path}")"""

##########################################################################################################################################################
##########################################################################################################################################################

"""def generate_patch_selection_report_pdf(
    config: dict,
    all_data,
    slide_mosaic_paths: dict,
    pdf_path: str,
    patch_size: int,
    thumb_size: int = 512,
    overview_grid: tuple = (2, 3),
    num_detail_patches: int = 10
) -> None:

    tile_px = config["benchmark_parameters"]["tile_px"]
    tile_size = tile_px[0] if isinstance(tile_px, (list, tuple)) else tile_px

    slide_ids = list(slide_mosaic_paths.keys())
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        visualize_patch_selection_overview(
            pdf,
            all_data,
            slide_ids,
            slide_mosaic_paths,
            thumb_size=thumb_size,
            tile_size=patch_size
        )

        visualize_patch_selection_details(
            pdf,
            slide_ids,
            slide_mosaic_paths,
            num_patches=num_detail_patches,
            patch_size=tile_size
        )

    logging.info(f"Combined PDF report written to: {pdf_path}")"""


