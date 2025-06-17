import logging 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import openslide
import os
import math

def generate_image_retrieval_report_by_class(
    results: list,
    all_data,
    output_dir: str,
    base_name: str = "retrieval_report",
    thumb_size: int = 512,
    per_file_limit: int = 200
):
    """
    Group retrieval results by query_label (class), and for each class
    produce one or more PDF(s) with at most `per_file_limit` pages each.

    Args:
        results (list of dict):
            Each dict must contain keys:
              - 'query_slide_id'
              - 'query_label'
              - 'predicted_label'
              - 'top_k' (a list of {slide_id, label, distance})
        all_data (sf.Dataset):
            Slideflow dataset, used to locate slide files by slide_id.
        output_path (str):
            If this ends in ".pdf", its dirname is used as the output folder,
            and its stem as the base name.  If it is a directory, PDFs
            will be written into that directory.
        thumb_size (int, optional):
            Pixel size of each thumbnail (default = 512).
        per_file_limit (int, optional):
            Maximum number of query‐slides per single PDF (default = 200).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Group results by query_label ---
    by_class = {}
    for r in results:
        lbl = r.get("query_label", "UNKNOWN")
        by_class.setdefault(lbl, []).append(r)

    # --- For each class, chunk into files of ≤ per_file_limit pages ---
    for query_label, slides_for_class in by_class.items():
        total = len(slides_for_class)
        n_parts = math.ceil(total / per_file_limit)

        for part_idx in range(n_parts):
            start = part_idx * per_file_limit
            end = min(start + per_file_limit, total)
            chunk = slides_for_class[start:end]

            # Build output filename: e.g. “<base_name>_<query_label>_part1.pdf”
            safe_label = str(query_label).replace(" ", "_")
            if n_parts == 1:
                out_pdf = os.path.join(output_dir, f"{base_name}_{safe_label}.pdf")
            else:
                out_pdf = os.path.join(
                    output_dir,
                    f"{base_name}_{safe_label}_part{part_idx + 1}.pdf"
                )

            logging.info(f"[{query_label}] Writing slides {start}–{end-1} "
                         f"of {total} → {out_pdf}")
            with PdfPages(out_pdf) as pdf:
                for result in chunk:
                    try:
                        fig = visualize_retrieval_result(result, all_data, thumb_size=thumb_size)
                        if fig is None:
                            continue
                        # Rasterize every image in the figure so PDF size stays small
                        for ax in fig.axes:
                            for im in ax.get_images():
                                im.set_rasterized(True)
                        pdf.savefig(fig, dpi=100, bbox_inches="tight")
                        plt.close(fig)
                    except Exception as e:
                        qid = result.get("query_slide_id", "<unknown>")
                        logging.warning(f"Visualization failed for {qid}: {e}")
            logging.info(f"Saved {out_pdf}")

    logging.info("All classes written.")

def visualize_retrieval_result(result, all_data, thumb_size=512):
    """
    Create a matplotlib figure showing the query slide and its top-k retrieved slides.

    Args:
        result (dict): Dictionary containing query and retrieval results.
        all_data (sf.Dataset): SlideFlow dataset used to look up slide paths.
        thumb_size (int): Thumbnail size for visualization.

    Returns:
        fig: Matplotlib figure object.
    """

    query_id = result['query_slide_id']
    query_label = result['query_label']
    predicted_label = result.get('predicted_label', None)
    top_k = result['top_k']
    k = len(top_k)

    # Determine grid size
    num_cols = 5
    num_rows = 0 if k == 0 else (2 if k > 5 else 1)

    # Create figure and layout grid
    fig = plt.figure(figsize=(15, 3 + num_rows * 3))
    gs = GridSpec(1 + num_rows, num_cols, figure=fig)

    # ---- Load and plot query slide ----
    query_slide_path = all_data.find_slide(slide=query_id)
    if query_slide_path is None:
        logging.warning(f"Could not find query slide: {query_id}")
        return None
    
    query_thumb = openslide.OpenSlide(query_slide_path).get_thumbnail((thumb_size, thumb_size))

    ax_query = fig.add_subplot(gs[0, :])
    ax_query.imshow(query_thumb)
    ax_query.set_title(f"Query: {query_id} (Label: {query_label})\nPredicted: {predicted_label}")
    ax_query.axis("off")

    # ---- Plot retrieved slides ----
    for i, hit in enumerate(top_k):
        row = 1 + i // num_cols
        col = i % num_cols
        slide_id = hit['slide_id']
        label = hit['label']
        distance = hit['distance']

        # Retrieve and check slide path
        slide_path = all_data.find_slide(slide=slide_id)
        if slide_path is None:
            logging.info(f"Slide not found: {slide_id}")
            continue

        thumb = openslide.OpenSlide(slide_path).get_thumbnail((thumb_size, thumb_size))

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(thumb)
        ax.set_title(f"{slide_id}\nLabel: {label}\nDist: {distance:.2f}", fontsize=8)
        ax.axis("off")

    # Adjust layout and return figure
    plt.tight_layout()
    return fig

def generate_image_retrieval_report_pdf(results, all_data, output_path):
    """
    Generate a PDF report of retrieval results (query + top-k), rasterizing images + reduced DPI.

    Args:
        results (list): Retrieval results, each with 'query_slide_id' and 'top_k' slides.
        all_data (sf.Dataset): Dataset used to locate the slides.
        pdf_path (str): Path to save the generated PDF.
    """

    # ---- Ensure output directory exists ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- Open PDF for writing pages ----
    with PdfPages(output_path) as pdf:
        for result in results:
            try:
                # Generate figure showing query and top-k slides
                fig = visualize_retrieval_result(result, all_data)
                if fig is None:
                    continue

                # Rasterize all image elements in the figure to reduce file size
                for ax in fig.axes:
                    for im in ax.get_images():
                        im.set_rasterized(True)

                # Save the figure as a page in the PDF
                pdf.savefig(fig, dpi=100, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                logging.info(f"Visualization failed for {result['query_slide_id']}: {e}")

    # ---- Final confirmation ----
    logging.info(f"Saved retrieval report PDF to: {output_path}")