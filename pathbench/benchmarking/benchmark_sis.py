# ==============================================================================
# Image Retrieval Benchmarking Pipeline
# ==============================================================================
# 
# This module implements a complete pipeline for benchmarking patch-based
# image retrieval in digital pathology. It includes functionality for:
# 
#   - Tile extraction using Slideflow and quality control filters
#   - Feature extraction using deep models
#   - Patch selection using SPLICE (RGB or features) or Yottixel (RGB or features)
#   - Visualization of selected patches and slide-level UMAP embeddings
#   - Leave-one-patient-out search benchmarking using Yottixel
#   - Evaluation of retrieval performance (hit@k, mmv@k, map@k)
#
# Supported patch selection methods:
#   - SPLICE: A streaming redundancy reduction algorithm (Alsaafin et al., 2024)
#   - Yottixel: Two-stage clustering for semantic and spatial diversity (Kalra et al., 2020)
#
# References:
#   - SPLICE: https://doi.org/10.48550/arXiv.2404.17704
#   - Yottixel: https://doi.org/10.1016/j.media.2020.101757
#
# Author: Jurre Weijer
# Project: PathBench-MIL (extension)
# Date: April 2025
# 
# Dependencies:
#   - Slideflow, PyTorch, OpenSlide, UMAP-learn
#   - NumPy, Pandas, Matplotlib, Seaborn, Joblib, TQDM, scikit-learn
#
# ==============================================================================

import os
import glob
import logging
from itertools import product
from matplotlib.style import available
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import torch
from umap.umap_ import UMAP
import openslide
import math
import sys
import importlib
import base64
import json
import joblib
from collections import defaultdict, Counter
import re
import random
from typing import List, Dict, Tuple
import multiprocessing as mp

import psutil, gc, tempfile, errno
import torch
import torch.multiprocessing as tmp

# Use file-backed tensors instead of /dev/shm
tmp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

from ..benchmarking.benchmark import generate_bags
from ..utils.utils import free_up_gpu_memory
from ..image_retrieval.config_validator import SISConfigValidator
from ..image_retrieval.utils import (load_patch_dicts_from_tfr, 
                                     load_patch_dicts_pickle, 
                                     save_patch_dicts_pickle, 
                                     save_retrieval_metrics, 
                                     save_retrieval_results_to_excel, 
                                     calculate_combinations,
                                     build_param_id_string,
                                     )
from ..image_retrieval.visualization.mosaic_selection import generate_mosaic_selection_report_pdf
from ..image_retrieval.evaluation import evaluate_retrieval_metrics, parse_metric_names
from ..image_retrieval.visualization.retrieval_results import generate_image_retrieval_report_pdf
from ..image_retrieval.visualization.umap import run_umap_visualizations
from ..image_retrieval.mosaic_selectors import splice, yottixel, sdm, hshr, pbms 
from ..image_retrieval.mosaic_selectors.registry import build_mosaic_selector
from ..image_retrieval.search_methods.registry import build_search_method, list_search_methods

import slideflow as sf
from slideflow.model import build_feature_extractor
from slideflow.slide import qc

logger = logging.getLogger(__name__)

def perform_tile_extraction(config, project, combination_dict):
    """
    Perform tile extraction for a given parameter combination in the project.
    
    This function sets up quality control (QC) methods based on configuration,
    then uses those methods to extract tiles from the whole slide images (WSIs)
    using Slideflow's built-in dataset handling.

    Args:
        config (dict): Configuration dictionary, including QC settings and filters.
        project (sf.Project): Slideflow project object.
        combination_dict (dict): Dictionary specifying current tile extraction parameters (e.g., tile size in px/um).

    Returns:
        all_data (sf.Dataset): Slideflow dataset object after tile extraction.
    """

    # ---- Setup Quality Control Methods ----
    qc_methods = config['experiment']['qc']
    qc_filters = config['experiment']['qc_filters']
    qc_list = []

    if qc_methods is not None:
        for qc_method in qc_methods:
            # Use CLAHE-enhanced Otsu if specified, otherwise load QC method by name
            if qc_method == 'Otsu-CLAHE':
                qc_list.append(getattr(qc, 'Otsu')(with_clahe=True))
            else:
                qc_list.append(getattr(qc, qc_method)())
        
        logging.info(f"Configured QC methods: {[type(m).__name__ for m in qc_list]}")
    else: 
        qc_list = None
        logging.info(f"Configured QC methods: {qc_list}")
    
    # ---- Pull out ROI parameters, with defaults ----
    roi_params = config['experiment'].get('roi_parameters', {})
    roi_method = roi_params.get('roi_method', 'auto')
    roi_filter  = roi_params.get('roi_filter_method', 'center')

    # ---- Load Dataset ----
    # Initializes or loads the Slideflow dataset object with the specified tile size
    all_data = project.dataset(tile_px=combination_dict['tile_px'],
                               tile_um=combination_dict['tile_um'])

    logging.info("Starting tile extraction...")

    # ---- Extract Tiles ----
    try:
        all_data.extract_tiles(
            enable_downsample=True,
            save_tiles=False,
            qc=qc_list,
            grayspace_fraction=float(qc_filters['grayspace_fraction']),
            whitespace_fraction=float(qc_filters['whitespace_fraction']),
            grayspace_threshold=float(qc_filters['grayspace_threshold']),
            whitespace_threshold=int(qc_filters['whitespace_threshold']),
            num_threads=config['experiment']['num_workers'],
            report=config['experiment']['report'],
            roi_method=roi_method,
            roi_filter_method=roi_filter
        )
    except Exception as e:
        logging.error(f"Tile extraction failed: {e}")
        raise

    return all_data

def perform_feature_extraction(config, project, all_data, combination_dict, string_without_mil):
    """
    Perform feature extraction on tiles using the specified feature extractor.

    This function builds a feature extractor based on the configuration and
    parameter combination, clears GPU memory to avoid OOM errors, and then 
    runs feature extraction using `generate_bags`.

    Args:
        config (dict): Configuration dictionary with experiment settings.
        project (sf.Project): Slideflow project object.
        all_data: Slideflow dataset containing extracted tiles.
        combination_dict (dict): Dictionary with current settings, including feature extractor name and tile size.
        string_without_mil (str): Identifier string used for saving paths (typically excluding '_mil').

    Returns:
        bags: Output of `generate_bags`
    """
    logging.info("Starting feature extraction...")

    # Free GPU memory before starting
    free_up_gpu_memory()

    # ---- Build feature extractor based on config ----
    feature_extractor = build_feature_extractor(
        name=combination_dict['feature_extraction'].lower(),
        tile_px=combination_dict['tile_px']
    )

    # ---- Run feature extraction ----
    bags = generate_bags(
        config=config,
        project=project,
        all_data=all_data,
        combination_dict=combination_dict,
        string_without_mil=string_without_mil,
        feature_extractor=feature_extractor
    )

    logging.info(f"Feature extraction completed successfully. Features stored at: {bags}")

    return bags

def make_patch_mosaic(args):
    """
    Worker function (runs in a separate process) to build a single slide's mosaic.
    Expects a tuple of:
      (slide_id, tfr_path, feats_pt, feats_idx, patches_pkl_folder,
       mosaic_folder, patch_size, method, percentile, config, selector_kwargs)
    """
    (slide_id, tfr_path, feats_pt, feats_idx,
     patches_pkl_folder, mosaic_folder,
     patch_size, selector_name, selector_params, config, selector_kwargs) = args  

    # ---- Paths to the **patch** dictionary dump ----
    patches_pkl = os.path.join(patches_pkl_folder, f"{slide_id}.pkl")

    # ---- 1) Load or build the patches pickle ----
    if os.path.exists(patches_pkl) and os.path.getsize(patches_pkl) > 0:
        patch_data = load_patch_dicts_pickle(patches_pkl, reconstruct_features=True)
    else:
        patch_data = load_patch_dicts_from_tfr(tfr_path, feats_idx, feats_pt, patch_size)
        if len(patch_data["patches"]) == 0:
            logging.error(f"[NO PATCHES] '{slide_id}' has no patches — skipping")
            return (slide_id, None, "no_patches")
        save_patch_dicts_pickle(patch_data, patches_pkl, compress=3)

    # ---- 2) Build the mosaic filename & run selection if needed ----
    mosaic_pkl     = os.path.join(mosaic_folder, f"{slide_id}.pkl")
    patch_ids_npz  = os.path.join(mosaic_folder, f"{slide_id}.npz")
    groups_json    = os.path.join(mosaic_folder, f"{slide_id}_groups.json")  # optional

    if not (os.path.exists(mosaic_pkl) and os.path.getsize(mosaic_pkl) > 0):
        patch_selector = build_mosaic_selector(
            selector_name, selector_params, config, **(selector_kwargs or {})
        )

        # Run selection (kwargs may be empty for methods without a param)
        selected, group_ids, coords, groups = patch_selector.run(patch_data["patches"], **{})

        subset = [patch_data["patches"][i] for i in selected]

        # let the selector contribute extra data (optional)
        try:
            additional_data = patch_selector.additional_data()
            if not isinstance(additional_data, dict):
                logging.warning(f"{selector_name}.additional_data did not return dict; ignoring.")
                additional_data = {}
        except Exception as e:
            logging.warning(f"additional_data failed for {slide_id}: {e}")
            additional_data = {}

        mosaic_dict = {
            "properties": patch_data["properties"],
            "patches": subset,
        }
        mosaic_dict.update(additional_data)

        save_patch_dicts_pickle(mosaic_dict, mosaic_pkl, compress=3)
        np.savez_compressed(patch_ids_npz, bin_ids=group_ids, coords=coords)

        try:
            groups_for_json = {int(g): [int(x) for x in idxs] for g, idxs in groups.items()}
            with open(groups_json, "w") as f:
                json.dump(groups_for_json, f)
        except Exception as e:
            logging.warning(f"Could not save groups for {slide_id}: {e}")

    return (slide_id, mosaic_pkl, None)

"""def make_slide_mosaic(
    args
):
    Build a “mosaic” pickle for a slide‐foundation model by pretending
    it’s just a single patch at (0,0).  Saves:

      {slide_id}.pkl  → {"properties":{"features_path":feats_pt},
                         "patches":[{"loc":(0,0), "feature_index":0}]}
      {slide_id}.npz  → bin_ids=[0], coords=[[0,0]]

    which exactly matches what your existing code expects.

    (slide_id, tfr_path, feats_pt, feats_idx,
     patches_pkl_folder, mosaic_folder,
     patch_size, selector_name, selector_params, config) = args
    
    # 1) Load the feature tensor
    feats = torch.load(feats_pt)
    arr   = feats.numpy() if hasattr(feats, "numpy") else feats
    arr   = np.asarray(arr)

    # if it’s 1-D, make it shape (1, F)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    # now arr.shape == (1, F)

    # 2) build the “patch_data” dict exactly like your multi‐patch pipelines
    patch_data = {
        "properties": {"tfr_path":            tfr_path,
                       "features_index_path": feats_idx,
                       "features_path":       feats_pt,},
        "patches": [
            {
                "tfr_index": None,
                "loc":       (0,0),
                "wsi_loc":   None,
                "feature":   arr[0],  # or patch_feature itself if you don't need to modify
                "rgb_histogram": None
            }
        ]
    }

    # 3) save .pkl via your existing save helper
    os.makedirs(mosaic_folder, exist_ok=True)
    pkl_path = os.path.join(mosaic_folder, f"{slide_id}.pkl")
    save_patch_dicts_pickle(patch_data, pkl_path, compress=3)

    # 4) also save the .npz so that your load paths never break
    npz_path = os.path.join(mosaic_folder, f"{slide_id}.npz")
    np.savez_compressed(npz_path,
                        bin_ids=np.array([0], dtype=int),
                        coords =np.array([[0, 0]], dtype=int))

    return slide_id, pkl_path, None"""

def create_slide_mosaic_mp(
    config,
    all_data,
    selector_name,
    selector_params,
    mosaics_base,
    features_folder_path,
    patch_size,
    **selector_kwargs
):
    logging.info(f"Running mosaic creation using method={selector_name}")

    slide_mosaic_paths = {}
    mosaic_failures = {}

    # Per-feature patches cache (leave as-is)
    patches_pkl_folder = os.path.join(
        mosaics_base,
        f"patches_{os.path.basename(os.path.normpath(features_folder_path))}"
    )
    os.makedirs(patches_pkl_folder, exist_ok=True)

    # Always suffix mosaics with features (keeps RGB/feature selectors consistent)
    features_suffix = f"_{os.path.basename(os.path.normpath(features_folder_path))}"

    # ID built only from true hyperparams
    param_id = build_param_id_string("selector", selector_name, selector_params, compact=True)
    params_suffix = f"_{param_id}" if param_id else ""

    mosaic_folder = os.path.join(
        mosaics_base,
        f"{selector_name}{params_suffix}{features_suffix}"
    )
    os.makedirs(mosaic_folder, exist_ok=True)

    # ---- derive in_dim without mutating selector_params ----
    first_sid = os.path.splitext(os.path.basename(next(iter(all_data.tfrecords()))))[0]
    first_feat = torch.load(os.path.join(features_folder_path, f"{first_sid}.pt"))
    in_dim = int(first_feat.shape[1])
    selector_kwargs = {**selector_kwargs, "in_dim": in_dim}

    # ---- schedule work ----
    to_process = []
    for tfr_path in tqdm(all_data.tfrecords(), desc="Scanning slides", file=sys.stdout):
        slide_id = os.path.splitext(os.path.basename(tfr_path))[0]
        feats_pt  = os.path.join(features_folder_path, f"{slide_id}.pt")
        feats_idx = os.path.join(features_folder_path, f"{slide_id}.index.npz")

        if not (os.path.exists(feats_pt) and os.path.exists(feats_idx)):
            raise FileNotFoundError(
                f"Missing features for slide {slide_id}: {feats_pt}, {feats_idx}"
            )

        mosaic_pkl = os.path.join(mosaic_folder, f"{slide_id}.pkl")
        if os.path.exists(mosaic_pkl) and os.path.getsize(mosaic_pkl) > 0:
            slide_mosaic_paths[slide_id] = mosaic_pkl
        else:
            # NOTE: include patches_pkl_folder here — matches worker’s expected args
            to_process.append((
                slide_id, tfr_path, feats_pt, feats_idx,
                patches_pkl_folder, mosaic_folder,
                patch_size, selector_name, selector_params, config, selector_kwargs
            ))

    if len(to_process) > 0:
        n_workers = min(len(to_process), max(1, mp.cpu_count() - 1))
        logging.info(f"Spawning {n_workers} workers to build {len(to_process)} mosaics...")
        with mp.Pool(n_workers) as pool:
            for slide_id, mosaic_pkl, failure in tqdm(
                pool.imap_unordered(make_patch_mosaic, to_process),
                total=len(to_process), desc="Building mosaics", file=sys.stdout
            ):
                if failure == "no_patches":
                    mosaic_failures.setdefault("no_patches", []).append(slide_id)
                elif mosaic_pkl is None:
                    logging.error(f"Worker returned no path for {slide_id}")
                else:
                    slide_mosaic_paths[slide_id] = mosaic_pkl

    if mosaic_failures:
        out_path = os.path.join(patches_pkl_folder, "roi_failures.json")
        with open(out_path, "w") as f:
            json.dump(mosaic_failures, f, indent=2)
        logging.info(f"Saved ROI failures for {len(mosaic_failures)} slides to {out_path}")

    return slide_mosaic_paths

def create_slide_feature_paths(
    config,
    all_data,
    features_folder_path: str,
    ext: str = ".pt",
    strict: bool = False
):
    slide_representation_paths = {}

    for tfr_path in tqdm(all_data.tfrecords(), desc="Scanning slides for features", file=sys.stdout):
        slide_id = os.path.splitext(os.path.basename(tfr_path))[0]  # no TFRecord open

        feat_path = os.path.join(features_folder_path, f"{slide_id}{ext}")
        if os.path.exists(feat_path) and os.path.getsize(feat_path) > 0:
            slide_representation_paths[slide_id] = feat_path
        else:
            logger.warning(f"Features are missing for slide {slide_id}")

    logger.info("Collected %d slide-level feature paths.", len(slide_representation_paths))
    return slide_representation_paths

def check_precomputed_features(all_data, features_folder_path):
    """
    Verify that all required precomputed features exist for the slides.

    Args:
        all_data: Slideflow dataset object (provides .tfrecords()).
        features_folder_path (str): Path to the folder containing .pt and .index.npz files.

    Returns:
        None

    Raises:
        FileNotFoundError: If one or more features are missing.
    """
    missing = []
    for tfr_path in all_data.tfrecords():
        slide_id = os.path.splitext(os.path.basename(tfr_path))[0]
        feats_pt  = os.path.join(features_folder_path, f"{slide_id}.pt")
        feats_idx = os.path.join(features_folder_path, f"{slide_id}.index.npz")
        if not (os.path.exists(feats_pt) and os.path.exists(feats_idx)):
            missing.append(slide_id)

    if missing:
        logging.error(f"Missing features for {len(missing)} slides: {', '.join(missing[:10])}"
                      f"{' ...' if len(missing) > 10 else ''}")
        raise FileNotFoundError("Precomputed features are missing. Cannot continue without GPU.")

def summarize_combination_strings(
    config: dict,
    combination_dict: dict,
) -> dict:
    """
    Return all canonical strings for a given combination.

    Output keys:
      - tile_string:      e.g. "256px_128um"
      - feature_string:   e.g. "256px_128um_reinhard_uni"
      - selector_name:    e.g. "splice_rgb" or "slide"
      - selector_param_id:e.g. "25" (compact, from include_in_id params only)
      - selector_full:    e.g. "splice_rgb_25" or "slide"
      - search_method:    e.g. "yottixel"
      - search_param_id:  e.g. "10" (compact, from include_in_id params only)
      - search_full:      e.g. "yottixel_10"
      - mosaic_string:    e.g. "splice_rgb_256px_128um_reinhard_uni"
      - combo_id:         e.g. "yottixel_10__splice_rgb_25__256px_128um_reinhard_uni"
                           (selector segment omitted for slide-level extractors)
    Notes:
      - Uses registry metadata via build_param_id_string("selector"| "search", name, params, compact=True).
      - Injects `mosaic_string` into search params (needed by e.g. SISH); it only appears in IDs
        if that param is marked `include_in_id=True` in the method HYPERPARAMS.
      - No heavy class instantiation.
    """
    # ---- basic feature strings ----
    tile_px = combination_dict["tile_px"]
    tile_um_raw = combination_dict["tile_um"]
    tile_um = str(tile_um_raw) if str(tile_um_raw).endswith("x") else f"{tile_um_raw}um"

    normalization = combination_dict["normalization"]
    extractor = combination_dict["feature_extraction"].lower()

    tile_string = f"{tile_px}px_{tile_um}"
    feature_string = f"{tile_string}_{normalization}_{extractor}"

    # ---- selector (only for patch-level extractors) ----
    validator = SISConfigValidator(config)
    is_patch = validator.is_patch_extractor(extractor)

    if is_patch:
        selector_name = combination_dict["mosaic_selector"].lower()
        selector_params = dict(combination_dict.get("mosaic_selector_params", {}) or {})
        selector_param_id = build_param_id_string("selector", selector_name, selector_params, compact=True)
        selector_full = f"{selector_name}_{selector_param_id}" if selector_param_id else selector_name
        mosaic_string = f"{selector_full}_{feature_string}"
    else:
        selector_name = "slide"
        selector_params = {}
        selector_param_id = ""
        selector_full = "slide"
        mosaic_string = f"slide_{feature_string}"

    # ---- search method (registry-driven, no instantiation) ----
    search_method = combination_dict["search_method"].lower()
    search_params = dict(combination_dict.get("search_method_params", {}) or {})
    search_param_id = build_param_id_string("search", search_method, search_params, compact=True)
    search_full = f"{search_method}_{search_param_id}" if search_param_id else search_method

    # ---- final combo id ----
    if is_patch:
        combo_id = f"{search_full}__{selector_full}__{feature_string}"
    else:
        combo_id = f"{search_full}__{feature_string}"

    return {
        "tile_string": tile_string,
        "feature_string": feature_string,
        "selector_name": selector_name,
        "selector_param_id": selector_param_id,
        "selector_full": selector_full,
        "search_method": search_method,
        "search_param_id": search_param_id,
        "search_full": search_full,
        "mosaic_string": mosaic_string,
        "combo_id": combo_id,
    }

def benchmark_sis(config, project):
    """
    Benchmarking for image retrieval experiments.

    This function calculates all parameter combinations, performs tile extraction,
    feature extraction, mosaic creation, and visualization.

    Args:
        config (dict): The configuration dictionary.
        project (sf.Project): The slideflow project.
    """
    logging.info("Starting image retrieval benchmarking...")

    # ---- Validate the user’s config before doing any work ----
    validator = SISConfigValidator(config)
    validator.validate()
    #logging.info("Configuration validated successfully, continuing with benchmarking.")

    # ---- Define paths ----
    project_dir = os.path.join("experiments", config['experiment']['project_name'])

    bags_base = os.path.join(project_dir, "bags")
    vis_base  = os.path.join(project_dir, "visualizations")
    eval_base = os.path.join(project_dir, "eval")
    mosaics_base = os.path.join(project_dir, "mosaics") 

    for path in (bags_base, vis_base, eval_base, mosaics_base):
        os.makedirs(path, exist_ok=True)

    # ---- Calculate parameter combinations ----
    all_combinations = calculate_combinations(config)
    logging.info(f"Total number of combinations: {len(all_combinations)}")

    resume_flag = bool(config['experiment'].get('resume', False))
    checkpoint_path = os.path.join(project_dir, 'completed_combinations.json')

    if resume_flag and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            completed = set(json.load(f))
        logging.info(f"Resuming from checkpoint with {len(completed)} completed combinations.")
    else:
        completed = set()
        if resume_flag:
            logging.warning("Resume=True but no checkpoint file found — starting from scratch.")
    
    # ---- Get visualization config ----
    visualization_cfg = config["experiment"].get("visualization", {})

    # ---- Iterate over each configuration ----
    for combination_dict in all_combinations:
        logging.info(f"Processing combination: {combination_dict}")

        if not validator.is_valid_combination(combination_dict):
            logging.warning(f"Skipping combo {combination_dict}: {validator.combination_error}")
            continue 

        id_strings = summarize_combination_strings(config, combination_dict)
        if id_strings["combo_id"] in completed:
            logging.info(f"Skipping already completed combo: {id_strings['combo_id']}")
            continue

        # ---- Tile extraction ----
        all_data = perform_tile_extraction(config, project, combination_dict)

        # ---- Feature extraction ----
        feature_string = id_strings["feature_string"]
        features_folder_path = os.path.join(bags_base, feature_string)

        if torch.cuda.is_available():
            bags = perform_feature_extraction(config, project, all_data, combination_dict, feature_string)
            del bags
        else:
            check_precomputed_features(all_data, features_folder_path) 
            logging.info("No GPU detected. Skipping feature extraction and using precomputed features.") 
        
        if config["experiment"].get("feature_extraction_only", False):
            logging.info("Feature extraction only mode enabled; skipping remaining steps.")
            continue

        # ---- Mosaic creation ----
        if validator.is_patch_extractor(combination_dict.get("feature_extraction")):
            mosaic_selector = combination_dict["mosaic_selector"].lower()
            selector_params = combination_dict.get("mosaic_selector_params", {})
            sel_with_params = id_strings["selector_full"] 

            logging.info(f"Running {mosaic_selector} patch selection...")
            slide_mosaic_paths = create_slide_mosaic_mp(
                                        config=config, 
                                        all_data=all_data,
                                        selector_name=mosaic_selector,
                                        selector_params=selector_params, 
                                        mosaics_base=mosaics_base,
                                        features_folder_path=features_folder_path, 
                                        patch_size=combination_dict['tile_px'], 
                                        prototypes_base=os.path.join(project_dir, "prototyping"),
                                        feature_string=feature_string,
                                    )

            slide_representation_paths = slide_mosaic_paths
            logging.info("Mosaic creation completed.")

            # ---- Patch visualization (multi-page PDF) ----
            logging.info("Creating mosaic patch visualizations...")

            pdf_base = os.path.join(vis_base, f"mosaics_{sel_with_params}_{feature_string}")

            if "patch_selection" in visualization_cfg:
                try:
                    generate_mosaic_selection_report_pdf(
                        config=config,
                        all_data=all_data,
                        slide_mosaic_paths=slide_representation_paths,
                        mosaic_selector=mosaic_selector,
                        pdf_base=pdf_base,
                        patch_px=combination_dict['tile_px'],
                        patch_um=combination_dict['tile_um']
                    )
                except Exception as e:
                    logging.warning(f"Patch visualization failed for {mosaic_selector}_{feature_string}: {e}")

            logging.info("Mosaic patch visualizations saved to PDF.")
            del slide_mosaic_paths
        elif validator.is_slide_extractor(combination_dict.get("feature_extraction")):
            slide_representation_paths = create_slide_feature_paths(config, all_data, features_folder_path)
            mosaic_selector = "slide"                
            sel_with_params = mosaic_selector        

        # ---- Generate UMAP plots (if requested) ----
        if "umap" in visualization_cfg:
            umap_base = os.path.join(vis_base, f"umap_{sel_with_params}_{feature_string}")
            umap_cfg = visualization_cfg["umap"]
            run_umap_visualizations(
                umap_cfg=umap_cfg,
                slide_representation_paths=slide_representation_paths,
                mosaic_selector=mosaic_selector,
                output_base=umap_base,
                annotation_file=config["experiment"]["annotation_file"],
                random_state=config["experiment"].get("random_state", None)
            )

        # ---- Similar Image Search Benchmark ----   
        search_method = combination_dict['search_method'].lower()
        search_params = combination_dict.get('search_method_params', {})
        search_params["mosaic_string"] = id_strings["mosaic_string"]

        searcher = build_search_method(
            name=search_method,
            config=config,
            slide_representation_paths=slide_representation_paths,
            params=search_params,
        )

        search_with_params = id_strings["search_full"] 

        logging.info(f"Running leave-one-patient-out evaluation using {search_method} retrieving {searcher.k} slides...")
        results = searcher.leave_one_patient_out()

        logging.info(f"Leave-one-patient-out completed with {len(results)} queries.")

        # ---- Save retrieval results ----
        combo_eval_folder = os.path.join(eval_base, f"{search_with_params}_{sel_with_params}_{feature_string}")
        os.makedirs(combo_eval_folder, exist_ok=True)

        save_retrieval_results_to_excel(
            results=results,
            output_path=os.path.join(combo_eval_folder, f"retrieval_results.xlsx")
        )

        if "retrieval_report" in visualization_cfg:
            try:
                generate_image_retrieval_report_pdf(
                    config=config,
                    results=results,
                    all_data=all_data,
                    output_dir=combo_eval_folder
                )
            except Exception as e:
                logging.warning(
                    f"Retrieval visualization failed for {search_with_params}_{sel_with_params}_{feature_string}: {e}"
                )

            logging.info("Image retrieval visualizations saved to PDF.")

        # ---- Metric Evaluation ----
        raw_metrics = config['experiment'].get("evaluation", []) or []
        valid_metrics = [m for m in raw_metrics if re.match(r'^(hit|mmv|map)_at_\d+$', m)]

        if not valid_metrics:
            logging.info("No valid evaluation metrics specified; skipping metric computation.")
        else:
            metric_results = evaluate_retrieval_metrics(results, valid_metrics)
            if metric_results:
                save_retrieval_metrics(config, metric_results, output_path=os.path.join(combo_eval_folder, f"retrieval_metrics.xlsx"))
            else:
                logging.info("Metric computation returned no results; skipping save.")
        
        completed.add(id_strings["combo_id"])
        with open(checkpoint_path,'w') as f:
            json.dump(sorted(completed), f, indent=2)

    logging.info("benchmark_sis completed.")