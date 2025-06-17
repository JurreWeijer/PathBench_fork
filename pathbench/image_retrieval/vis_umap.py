import logging
import pandas as pd
from umap.umap_ import UMAP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..image_retrieval.utils import load_patch_dicts_pickle

def plot_slide_umap(config, slide_mosaic_paths, mosaic_method, aggregation_method, umap_params, output_path):
    """
    Generate and save a UMAP plot from slide- or patch-level features.

    - Patch-level (aggregation_method=None): plots every patch individually.
    - Slide-level ("mean" or "median"): aggregates features per slide.

    Args:
        config:          Experiment config with annotation CSV under
                         config['experiment']['annotation_file'].
        slide_mosaic_paths: Mapping from slide_id to the path of its .pkl.
        mosaic_method:   Name of the patch-selection method (for titles).
        aggregation_method: "mean", "median", or None.
        umap_params:     Dict of UMAP args (n_neighbors, min_dist, metric, etc).
        output_path:     Where to save the resulting PNG.
    """

    # load slide labels
    ann = pd.read_csv(config['experiment']['annotation_file']).set_index('slide')
    slide_feats, slide_labels = [], []

    for slide_id, mosaic_pkl in slide_mosaic_paths.items():
        # get class label
        try:
            label = ann.loc[slide_id]["category"]
        except KeyError:
            logging.warning(f"no annotation for {slide_id}, skipping UMAP")
            continue

        # load only the selected patch dicts (which re-inserts .feature)
        mosaic_data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=True)
        mosaic_patches = mosaic_data["patches"]

        # pull out the feature vectors
        feats = np.stack([p['feature'] for p in mosaic_patches], axis=0)

        # aggregate or not
        if aggregation_method is None or aggregation_method.lower()=="none":
            slide_feats.extend(feats)
            slide_labels.extend([label]*len(feats))
        elif aggregation_method=="mean":
            slide_feats.append(feats.mean(axis=0))
            slide_labels.append(label)
        elif aggregation_method=="median":
            slide_feats.append(np.median(feats, axis=0))
            slide_labels.append(label)
        else:
            raise ValueError(f"bad aggregation {aggregation_method}")

    if not slide_feats:
        logging.error("no features collected for UMAP")
        return

    slide_feats = np.array(slide_feats)
    reducer = UMAP(
        n_neighbors= umap_params.get("n_neighbors",15),
        min_dist= umap_params.get("min_dist",0.1),
        metric= umap_params.get("metric","euclidean"),
        random_state= config['experiment'].get("random_state", None)
    )
    emb = reducer.fit_transform(slide_feats)

    plt.figure(figsize=(10,8))
    sns.scatterplot(
        x=emb[:,0], y=emb[:,1],
        hue=slide_labels, palette="tab10",
        s=80, alpha=0.8
    )
    plt.title(f"UMAP ({aggregation_method}) — {mosaic_method}")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_umap_visualizations(config, slide_mosaic_paths, mosaic_method, output_base):
    """
    Run UMAP visualizations for each configured aggregation method.

    Iterates over the entries in config['visualization'] that start with "UMAP",
    parses the aggregation method from the name (e.g., "UMAP-mean"), and
    triggers UMAP generation for each. Falls back to "mean" if unspecified.

    Args:
        config (dict): Experiment configuration dictionary.
        slide_mosaics (dict): Mapping from slide_id to list of patch dictionaries.
        mosaic_method (str): Name of the patch selection method.
        save_string (str): Identifier string used in output filenames.
    """
    vizs = config.get("visualization", [])
    # flatten umap_parameters into a dict
    umap_params = { list(d.keys())[0]: list(d.values())[0] for d in config.get("umap_parameters",[]) }

    for viz in vizs:
        if not viz.startswith("UMAP"):
            continue
        # parse "UMAP" or "UMAP-mean"/"UMAP-none"
        parts = viz.split("-",1)
        agg = None if len(parts)==1 or parts[1].lower()=="none" else parts[1]
        output_path = f"{output_base}_{agg}.png" if agg else f"{output_base}_none.png"
        try:
            logging.info(f"Running UMAP {agg}...")
            plot_slide_umap(
                config=config,
                slide_mosaic_paths=slide_mosaic_paths,
                mosaic_method=mosaic_method,
                aggregation_method=agg,
                umap_params=umap_params,
                output_path=output_path,
            )
        except Exception as e:
            logging.warning(f"UMAP {viz} failed: {e}")

        logging.info(f"Saved UMAP to {output_path}")
        
    