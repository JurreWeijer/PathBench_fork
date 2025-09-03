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
from typing import List, Dict
import multiprocessing as mp

import psutil, gc, tempfile, errno
import torch
import torch.multiprocessing as tmp

# Use file-backed tensors instead of /dev/shm
tmp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

from ..benchmarking.benchmark import calculate_combinations, generate_bags
from ..utils.utils import free_up_gpu_memory
import pathbench.image_retrieval.patch_selection as patch_selection_module
from ..image_retrieval.yottixel_search import YottixelDatabase
from ..image_retrieval.retccl_search import RetCCLDatabase
from ..image_retrieval.sish_search import SISHDatabase
from ..image_retrieval.config_validator import SISConfigValidator
from ..image_retrieval.utils import load_patch_dicts_from_tfr, load_patch_dicts_pickle, save_patch_dicts_pickle, save_retrieval_metrics, save_retrieval_results_to_excel, log_mem, cleanup_dev_shm
from ..image_retrieval.vis_patch_selection import generate_patch_selection_report_pdf
from ..image_retrieval.evaluation import evaluate_retrieval_metrics, parse_metric_names
from ..image_retrieval.vis_retrieval_results import generate_image_retrieval_report_pdf
from ..image_retrieval.vis_umap import run_umap_visualizations

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
            roi_method= roi_method,
            roi_filter_method= roi_filter,
            max_downsample = 4
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

def make_patch_mosaic(
    args
):
    """
    Worker function (runs in a separate process) to build a single slide's mosaic.
    Expects a tuple of:
      (slide_id, tfr_path, feats_pt, feats_idx, patches_pkl_folder,
       mosaic_folder, roi, patch_size, method, percentile, config)
    """
    (slide_id, tfr_path, feats_pt, feats_idx,
     patches_pkl_folder, mosaic_folder,
     patch_size, method, percentile, config) = args

    # --- Resolve selection function ---
    method_fn = f"{method.lower()}_patch_selection"
    patch_selection_fn = getattr(patch_selection_module, method_fn)

    # ---- Paths to the **patch** dictionary dump ----
    patches_pkl = os.path.join(patches_pkl_folder, f"{slide_id}.pkl")

    # ---- 1) Load or build the patches pickle ----
    if os.path.exists(patches_pkl) and os.path.getsize(patches_pkl) > 0:
        patch_data = load_patch_dicts_pickle(patches_pkl, reconstruct_features=True)
    else:
        patch_data = load_patch_dicts_from_tfr(tfr_path, feats_idx, feats_pt, patch_size)

        # If STILL zero patches, skip entirely
        if len(patch_data["patches"]) == 0:
            logging.error(f"[NO PATCHES] '{slide_id}' has no patches — skipping")
            return (slide_id, None, "no_patches")

        # Save the patch dump for future invocations
        save_patch_dicts_pickle(patch_data, patches_pkl, compress=3)

    # ---- 2) Build the mosaic filename & run selection if needed ----
    mosaic_pkl     = os.path.join(mosaic_folder, f"{slide_id}.pkl")
    patch_ids_npz  = os.path.join(mosaic_folder, f"{slide_id}.npz")
    groups_json    = os.path.join(mosaic_folder, f"{slide_id}_groups.json")  # optional

    if not (os.path.exists(mosaic_pkl) and os.path.getsize(mosaic_pkl) > 0):
        # NEW: selectors now return 4 items
        selected, group_ids, coords, groups = patch_selection_fn(
            config, patch_data["patches"], percentile
        )

        subset = [patch_data["patches"][i] for i in selected]

        save_patch_dicts_pickle(
            {"properties": patch_data["properties"], "patches": subset},
            mosaic_pkl,
            compress=3
        )

        # Keep existing key names so downstream code doesn’t break
        np.savez_compressed(patch_ids_npz, bin_ids=group_ids, coords=coords)

        # Optional but recommended: persist groups for later visualization/QA
        try:
            # convert np.ndarray -> list for JSON
            groups_for_json = {int(g): [int(x) for x in idxs] for g, idxs in groups.items()}
            with open(groups_json, "w") as f:
                json.dump(groups_for_json, f)
        except Exception as e:
            logging.warning(f"Could not save groups for {slide_id}: {e}")

    return (slide_id, mosaic_pkl, None)

def make_slide_mosaic(
    args
):
    """
    Build a “mosaic” pickle for a slide‐foundation model by pretending
    it’s just a single patch at (0,0).  Saves:

      {slide_id}.pkl  → {"properties":{"features_path":feats_pt},
                         "patches":[{"loc":(0,0), "feature_index":0}]}
      {slide_id}.npz  → bin_ids=[0], coords=[[0,0]]

    which exactly matches what your existing code expects.
    """

    (slide_id, tfr_path, feats_pt, feats_idx,
     patches_pkl_folder, mosaic_folder,
     patch_size, method, percentile, config) = args
    
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

    return slide_id, pkl_path, None

def create_slide_mosaic_mp(
    config,
    all_data,
    method,
    percentile,
    mosaics_base,
    features_folder_path,
    patch_size
):
    """
    Generate patch-mosaics for each slide, skipping slides whose mosaic already exists,
    then parallelizing the remaining work across CPU cores.

    Returns:
        dict[slide_id → path_to_mosaic_pkl]
    """
    logging.info(f"Running mosaic creation using method={method}")

    # 1) Prepare all the base‐paths/folders
    slide_mosaic_paths = {}
    mosaic_failures = {}

    patches_pkl_folder = os.path.join(mosaics_base, f"patches_{os.path.basename(os.path.normpath(features_folder_path))}") #TODO: see if we can store the patches only once
    os.makedirs(patches_pkl_folder, exist_ok=True)

    pct_str = "all" if percentile is None else str(percentile) #TODO: all should not be added when there is no percentage
    features_suffix = (
        f"_{os.path.basename(os.path.normpath(features_folder_path))}"
        if "features" in method.lower() else ""
    )
    mosaic_folder = os.path.join(
        mosaics_base,
        f"{method}_{pct_str}{features_suffix}"
    )
    os.makedirs(mosaic_folder, exist_ok=True)

    # 2) First pass: classify each slide as "already done" vs. "needs work"
    to_process = []
    for tfr_path in tqdm(all_data.tfrecords(), desc="Scanning slides", file=sys.stdout):
        slide_id = os.path.splitext(os.path.basename(tfr_path))[0]

        feats_pt  = os.path.join(features_folder_path, f"{slide_id}.pt")
        feats_idx = os.path.join(features_folder_path, f"{slide_id}.index.npz")

        # Must exist on disk
        if not os.path.exists(feats_pt) or not os.path.exists(feats_idx):
            raise FileNotFoundError(
                f"Missing features for slide {slide_id}: {feats_pt}, {feats_idx}"
            )

        # Already‐computed mosaic?
        mosaic_pkl = os.path.join(mosaic_folder, f"{slide_id}.pkl")
        if os.path.exists(mosaic_pkl) and os.path.getsize(mosaic_pkl) > 0:
            # we can immediately register it as done
            slide_mosaic_paths[slide_id] = mosaic_pkl
        else:
            # we’ll need to build everything for this slide in the worker step
            to_process.append(
                (
                    slide_id,
                    tfr_path,
                    feats_pt,
                    feats_idx,
                    patches_pkl_folder,
                    mosaic_folder,
                    patch_size,
                    method,
                    percentile,
                    config
                )
            )

    # 3) If there are slides to process, spawn a Pool
    if len(to_process) > 0:
        n_workers = min(len(to_process), max(1, mp.cpu_count() - 1))
        logging.info(f"Spawning {n_workers} workers to build {len(to_process)} mosaics...")
        with mp.Pool(n_workers) as pool:
            for slide_id, mosaic_pkl, failure in tqdm(pool.imap_unordered(make_patch_mosaic, to_process), total=len(to_process), desc="Building mosaics", file=sys.stdout):
                if failure == "no_patches":
                    mosaic_failures.setdefault("no_patches", []).append(slide_id)
                elif mosaic_pkl is None:
                    # Shouldn’t really happen, but just in case
                    logging.error(f"Worker returned no path for {slide_id}")
                else:
                    slide_mosaic_paths[slide_id] = mosaic_pkl

    # 4) Write out ROI‐failure report if any
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
    logging.info("Configuration validated successfully, continuing with benchmarking.")

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
    benchmark_parameters = config['benchmark_parameters']

    resume_mode = config['experiment'].get('resume','from_beginning')
    checkpoint_path = os.path.join(project_dir, 'completed_combinations.json')
    if resume_mode == 'continue' and os.path.exists(checkpoint_path):
        with open(checkpoint_path,'r') as f:
            completed = set(json.load(f))
    else:
        completed = set()

    # ---- Iterate over each configuration ----
    for combination in all_combinations:
        combination_dict = {param: value for param, value in zip(benchmark_parameters.keys(), combination)}
        logging.info(f"Processing combination: {combination_dict}")

        # Strings used for filenames and identifiers
        tile_string = f"{combination_dict['tile_px']}px_{str(combination_dict['tile_um']) if str(combination_dict['tile_um']).endswith('x') else str(combination_dict['tile_um']) + 'x'}"
        feature_string = "_".join([f"{value}" for key, value in combination_dict.items() if key not in ['mil', 'loss', 'augmentation', 'activation_function', 'optimizer', 'mosaic_method', 'search_method', 'roi']])
        combo_id = "_".join([f"{value}" for key, value in combination_dict.items()])

        if not validator.is_valid_combination(combination_dict):
            logging.warning(f"Skipping combo {combo_id!r}: {validator.combination_error}")
            continue 

        if combo_id in completed:
            logging.info(f"Skipping already completed combo: {combo_id}")
            continue

        # ---- Tile extraction ----
        all_data = perform_tile_extraction(config, project, combination_dict)

        # ---- Feature extraction ----
        features_folder_path = os.path.join(bags_base, feature_string)

        if torch.cuda.is_available():
            bags = perform_feature_extraction(config, project, all_data, combination_dict, feature_string)
            del bags
        else:
            check_precomputed_features(all_data, features_folder_path) 
            logging.info("No GPU detected. Skipping feature extraction and using precomputed features.") 

        # ---- Mosaic creation ----
        if validator.is_patch_model(combination_dict.get("feature_extraction")):
            mosaic = combination_dict['mosaic_method']
            if "-" in mosaic:
                mosaic_method, mosaic_percentile = mosaic.split("-")
                mosaic_percentile = None if mosaic_percentile.lower() == "none" else int(mosaic_percentile)
            else:
                mosaic_method = mosaic
                mosaic_percentile = None

            logging.info(f"Running {mosaic_method} patch selection...")
            slide_mosaic_paths = create_slide_mosaic_mp(
                                        config=config, 
                                        all_data=all_data,
                                        method=mosaic_method, 
                                        percentile=mosaic_percentile, 
                                        mosaics_base=mosaics_base,
                                        features_folder_path=features_folder_path, 
                                        patch_size=combination_dict['tile_px'], 
                                    )

            slide_representation_paths = slide_mosaic_paths
            logging.info("Mosaic creation completed.")

            # ---- Patch visualization (multi-page PDF) ----
            logging.info("Creating mosaic patch visualizations...")

            if "features" in mosaic_method:
                pdf_base = os.path.join(vis_base, f"mosaics_{mosaic_method}_{tile_string}_{feature_string}")
            else:
                pdf_base = os.path.join(vis_base, f"mosaics_{mosaic_method}_{tile_string}")

            if config['experiment']['report']:
                try:
                    generate_patch_selection_report_pdf(
                        config=config,
                        all_data=all_data,
                        slide_mosaic_paths=slide_representation_paths,
                        mosaic_method=mosaic_method,
                        pdf_base=pdf_base,
                        patch_px=combination_dict["tile_px"],
                        patch_um=combination_dict['tile_um']
                    )
                except Exception as e:
                    logging.warning(f"Patch visualization failed for {mosaic_method}_{feature_string}: {e}")

            logging.info("Mosaic patch visualizations saved to PDF.")
            del slide_mosaic_paths
        elif validator.is_slide_model(combination_dict.get("feature_extraction")):
            slide_representation_paths = create_slide_feature_paths(config, all_data, features_folder_path)
            mosaic_method = "slide"
            logging.info(f"Found slide-level features for {len(slide_representation_paths)} slides in {features_folder_path}")

        # ---- Generate UMAP plots (if requested) ----
        umap_base = os.path.join(vis_base, f"umap_{mosaic_method}_{feature_string}")

        if any(viz.startswith("UMAP") for viz in config["experiment"].get("visualization", [])):
            run_umap_visualizations(
                config=config, 
                slide_representation_paths=slide_representation_paths, 
                mosaic_method=mosaic_method, 
                output_base=umap_base
            )

        # ---- Similar Image Search Benchmark ----
        search_string = combination_dict.get('search_method', 'yottixel-10')
        search_parts = search_string.split('-')
        search_method = search_parts[0]
        k = int(search_parts[1]) if len(search_parts) > 1 else 5

        logging.info(f"Running leave-one-patient-out evaluation using {search_method} retrieving {k} slides...")
        # Run search method
        if search_method.lower() == 'yottixel':
            search_database = YottixelDatabase(config=config, slide_representation_paths=slide_representation_paths, k=k)
            results = search_database.leave_one_patient_out()
        elif search_method.lower() == 'sish':
            search_database = SISHDatabase(config=config, slide_representation_paths=slide_representation_paths, k=k, mosaic_string=f"{mosaic_method}_{feature_string}")
            results = search_database.leave_one_patient_out()
        elif search_method.lower() == 'retccl':
            search_database = RetCCLDatabase(config=config, slide_representation_paths=slide_representation_paths, k=k)
            results = search_database.leave_one_patient_out()
        else:
            raise ValueError(f"Search method '{search_method}' is not implemented. Please choose a supported method.")
        
        logging.info(f"Leave-one-patient-out completed with {len(results)} queries.")

        # ---- Save retrieval results ----
        combo_eval_folder = os.path.join(eval_base, f"{search_method}_{mosaic_method}_{feature_string}")
        os.makedirs(combo_eval_folder, exist_ok=True)

        save_retrieval_results_to_excel(
            results=results,
            output_path=os.path.join(combo_eval_folder, f"retrieval_results.xlsx")
        )

        if config['experiment']['report']:
            try:
                generate_image_retrieval_report_pdf(
                    config=config,
                    results=results,
                    all_data=all_data,
                    output_dir=combo_eval_folder,   
                )
            except Exception as e:
                logging.warning(f"Retrieval visualization failed for {search_method}_{mosaic_method}_{feature_string}: {e}")

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
        
        completed.add(combo_id)
        with open(checkpoint_path,'w') as f:
            json.dump(sorted(completed), f, indent=2)

    logging.info("benchmark_sis completed.")