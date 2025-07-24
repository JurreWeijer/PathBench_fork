import logging
import pandas as pd
from umap.umap_ import UMAP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import gc
import psutil

from ..image_retrieval.utils import load_patch_dicts_pickle

def plot_slide_umap(config, slide_representation_paths, mosaic_method, aggregation_method, umap_params, output_path):
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

    for slide_id, repr_path in slide_representation_paths.items():
        # get class label
        try:
            label = ann.loc[slide_id]["category"]
        except KeyError:
            logging.warning(f"no annotation for {slide_id}, skipping UMAP")
            continue

        # load only the selected patch dicts (which re-inserts .feature)
        if repr_path.endswith(".pkl"):
            data = load_patch_dicts_pickle(repr_path, reconstruct_features=True)
            feats = np.stack([p['feature'] for p in data["patches"]], axis=0)
        elif repr_path.endswith(".pt"):
            feats_raw = torch.load(repr_path)
            feats = feats_raw.cpu().numpy()
        else:
            logging.warning(f"unrecognized extension for {repr_path}, skipping")
            continue
        
        # aggregate or not
        if aggregation_method is None:
            slide_feats.extend(feats)
            slide_labels.extend([label]*len(feats))
        elif aggregation_method == "mean":
            slide_feats.append(feats.mean(axis=0))
            slide_labels.append(label)
        elif aggregation_method == "median":
            slide_feats.append(np.median(feats, axis=0))
            slide_labels.append(label)
        else:
            raise ValueError(f"bad aggregation {aggregation_method}")

    if not slide_feats:
        logging.error("no features collected for UMAP")
        return

    X = np.vstack(slide_feats)
    reducer = UMAP(
        n_neighbors= umap_params.get("n_neighbors",15),
        min_dist= umap_params.get("min_dist",0.1),
        metric= umap_params.get("metric","euclidean"),
        random_state= config['experiment'].get("random_state", None)
    )
    emb = reducer.fit_transform(X)
    
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

def run_umap_visualizations(config, slide_representation_paths, mosaic_method, output_base):
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
    exts = {os.path.splitext(p)[1] for p in slide_representation_paths.values()}
    if len(exts) > 1:
        logging.error(f"UMAP: mixed representation types found ({exts}); cannot proceed")
        return
    is_slide = exts == {'.pt'}

    # flatten umap_parameters into a dict
    umap_params = { list(d.keys())[0]: list(d.values())[0] for d in config.get("umap_parameters",[]) }

    vizs = config["experiment"].get("visualization", [])
    aggs_methods = []
    for viz in vizs:
        if not viz.startswith("UMAP"):
            continue
        # parse "UMAP" or "UMAP-mean"/"UMAP-none"
        parts = viz.split("-",1)
        agg = None if len(parts)==1 or parts[1].lower()=="none" else parts[1]
        aggs_methods.append(agg)
    if not aggs_methods: 
        return
    
    # if slide-level, only unaggregated makes sense
    if is_slide:
        aggs_methods = [None] 
        
    for agg in aggs_methods:
        suffix = agg if agg else "none"
        output_path = f"{output_base}_{suffix}.png"
        try:
            logging.info(f"Running UMAP {agg}...")
            plot_slide_umap(
                config=config,
                slide_representation_paths=slide_representation_paths,
                mosaic_method=mosaic_method,
                aggregation_method=agg,
                umap_params=umap_params,
                output_path=output_path,
            )
        except Exception as e:
            logging.warning(f"UMAP {viz} failed: {e}")

        logging.info(f"Saved UMAP to {output_path}")
        
    # ---- CLEANUP ----
    # Drop large locals so GC can reclaim them
    for name in ["X", "emb", "slide_feats", "slide_labels", "feats",
                 "feats_raw", "data"]:
        if name in locals():
            del locals()[name]

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()