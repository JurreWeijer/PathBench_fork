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

from ..benchmarking.benchmark import calculate_combinations, generate_bags
from ..utils.utils import free_up_gpu_memory
import pathbench.image_retrieval.patch_selection as patch_selection_module
from ..image_retrieval.yottixel_search import YottixelDatabase
from ..image_retrieval.sish_search import SISHDatabase
from ..image_retrieval.config_validator import SISConfigValidator
from ..image_retrieval.utils import load_patch_dicts_from_tfr, load_patch_dicts_pickle, save_patch_dicts_pickle, save_retrieval_metrics, save_retrieval_results_to_excel
from ..image_retrieval.vis_patch_selection import generate_extensive_patch_selection_report_pdf, generate_simple_patch_selection_report_pdf
from ..image_retrieval.evaluation import evaluate_retrieval_metrics, parse_metric_names
from ..image_retrieval.vis_retrieval_results import generate_image_retrieval_report_by_class, generate_image_retrieval_report_pdf
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
        bags: Output of `generate_bags` #TODO: what is the output?
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

def create_slide_mosaic(config, all_data, method, percentile, mosaics_base, features_folder_path, patch_size):
    """
    Generate patch mosaics for each slide using a specified patch selection method.

    For each slide:
      - If both the full-patch PKL and the mosaic PKL already exist, skips processing.
      - Otherwise, loads or builds the full-patch dictionary (from TFRecord and feature files).
      - Applies the chosen patch selection function to pick a subset of patches.
      - Saves the selected subset as a new mosaic PKL.

    Args:
        config (dict):
            Experiment configuration dictionary.
        all_data (sf.Dataset):
            Slideflow dataset object providing TFRecord file paths.
        method (str):
            Name of the patch selection method to apply (e.g. "splice_rgb", "yottixel").
        percentile (float or None):
            Selection threshold for the method (e.g. percentile for SPLICE/Yottixel);
            use `None` to select all patches.
        mosaics_base (str):
            Directory under which full-patch PKLs are stored (must already exist).
        features_folder_path (str):
            Directory containing per-slide feature files (.pt and .index.npz).
        patch_size (int):
            Size of each tile in pixels, used for spatial matching against feature indices.

    Returns:
        dict:
            A mapping from slide ID (str) to the filepath (str) of the saved mosaic PKL
            containing only the selected patch subset.
    """

    # resolve the function as before…
    method_fn = f"{method.lower()}_patch_selection"
    patch_selection_fn = getattr(patch_selection_module, method_fn)

    slide_mosaic_paths = {}
    mosaic_failures = {}

    # append roi‐flag to filename only
    patches_pkl_folder = os.path.join(mosaics_base, f"patches")
    os.makedirs(patches_pkl_folder, exist_ok=True)

    pct_str = "all" if percentile is None else str(percentile)
    features_suffix = f"_{os.path.basename(os.path.normpath(features_folder_path))}" if "features" in method.lower() else ""
    mosaic_folder = os.path.join(mosaics_base, f"{method}_{pct_str}{features_suffix}")
    os.makedirs(mosaic_folder, exist_ok=True)

    for tfr_path in tqdm(all_data.tfrecords(), desc="Processing slides", file=sys.stdout):
        slide_id = sf.TFRecord(tfr_path)[0].get("slide", os.path.basename(tfr_path))

        # ---- Paths to the **patch** dictionary dump ----
        patches_pkl = os.path.join(patches_pkl_folder, f"{slide_id}.pkl")

        # ---- Paths to the **feature** files ----
        feats_pt  = os.path.join(features_folder_path, f"{slide_id}.pt")
        feats_idx = os.path.join(features_folder_path, f"{slide_id}.index.npz")

        if not os.path.exists(feats_pt) or not os.path.exists(feats_idx):
            raise FileNotFoundError(f"Missing features for slide {slide_id}: {feats_pt}, {feats_idx}")

        # ---- Load or build the patches pickle ----
        if os.path.exists(patches_pkl) and os.path.getsize(patches_pkl) > 0:
            patch_data = load_patch_dicts_pickle(patches_pkl, reconstruct_features=True)
        else:
            # First try with ROI if requested
            patch_data = load_patch_dicts_from_tfr(tfr_path, feats_idx, feats_pt, patch_size)

            if len(patch_data["patches"]) == 0:
                logging.error(f"[NO PATCHES] slide '{slide_id}' has no patches even without ROI - skipping")
                mosaic_failures.setdefault("no patches without ROI", []).append(slide_id)
                continue

            # save patch dump
            save_patch_dicts_pickle(patch_data, patches_pkl, compress=3)

        # ---- Now pick your mosaic filename, including the same suffix ----
        mosaic_pkl = os.path.join(mosaic_folder, f"{slide_id}.pkl")

        # ---- Run selection if needed ----
        if not (os.path.exists(mosaic_pkl) and os.path.getsize(mosaic_pkl) > 0):
            selected = patch_selection_fn(config, patch_data["patches"], percentile)
            subset  = [patch_data["patches"][i] for i in selected]
            save_patch_dicts_pickle({"properties": patch_data["properties"], "patches": subset}, mosaic_pkl, compress=3)

        slide_mosaic_paths[slide_id] = mosaic_pkl  # key by base_id, value is the correct ROI vs no-ROI file
    
    # ---- write ROI failures out to disk ----
    if mosaic_failures:
        out_path = os.path.join(patches_pkl_folder, "roi_failures.json")
        with open(out_path, 'w') as f:
            json.dump(mosaic_failures, f, indent=2)

        logging.info(f"Saved ROI failures for {len(mosaic_failures)} slides to {out_path}")

    return slide_mosaic_paths

def make_mosaic(
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
        # First attempt: use ROI if requested
        patch_data = load_patch_dicts_from_tfr(tfr_path, feats_idx, feats_pt, patch_size)

        # If STILL zero patches, skip entirely
        if len(patch_data["patches"]) == 0:
            logging.error(f"[NO PATCHES] '{slide_id}' has no patches — skipping")
            return (slide_id, None, "no_patches")

        # Save the patch dump for future invocations
        save_patch_dicts_pickle(patch_data, patches_pkl, compress=3)

    # ---- 2) Build the mosaic filename & run selection if needed ----
    mosaic_pkl = os.path.join(mosaic_folder, f"{slide_id}.pkl")

    if not (os.path.exists(mosaic_pkl) and os.path.getsize(mosaic_pkl) > 0):
        selected = patch_selection_fn(config, patch_data["patches"], percentile)
        subset = [patch_data["patches"][i] for i in selected]

        save_patch_dicts_pickle(
            {"properties": patch_data["properties"], "patches": subset},
            mosaic_pkl,
            compress=3
        )

    return (slide_id, mosaic_pkl, None)

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
        slide_id = sf.TFRecord(tfr_path)[0].get("slide", os.path.basename(tfr_path))

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
            for slide_id, mosaic_pkl, failure in tqdm(pool.imap_unordered(make_mosaic, to_process), total=len(to_process), desc="Building mosaics", file=sys.stdout):
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

        # Set default values for missing keys
        combination_dict.setdefault("roi", False)

        # Strings used for filenames and identifiers
        tile_string = f"{combination_dict['tile_px']}px_{str(combination_dict['tile_um']) if str(combination_dict['tile_um']).endswith('x') else str(combination_dict['tile_um']) + 'x'}"
        feature_string = "_".join([f"{value}" for key, value in combination_dict.items() if key not in ['mil', 'loss', 'augmentation', 'activation_function', 'optimizer', 'mosaic_method', 'search_method', 'roi']])
        combo_id = "_".join([f"{value}" for key, value in combination_dict.items()])

        if combo_id in completed:
            logging.info(f"Skipping already completed combo: {combo_id}")
            continue

        # ---- Tile extraction ----
        all_data = perform_tile_extraction(config, project, combination_dict)
        if config['experiment'].get('tile_extraction_only', False):
            logging.info("Tile extraction only mode enabled. Exiting after tile extraction.")
            continue

        # ---- Feature extraction ----
        bags = perform_feature_extraction(config, project, all_data, combination_dict, feature_string)
        if config['experiment'].get('feature_extraction_only', False):
            logging.info("Feature extraction only mode enabled. Exiting after feature extraction.")
            continue

        # ---- Mosaic creation ----
        mosaic = combination_dict['mosaic_method']
        if "-" in mosaic:
            mosaic_method, mosaic_percentile = mosaic.split("-")
            mosaic_percentile = None if mosaic_percentile.lower() == "none" else int(mosaic_percentile)
        else:
            mosaic_method = mosaic
            mosaic_percentile = None

        logging.info(f"Running {mosaic_method} patch selection...")
        features_folder_path = os.path.join(bags_base, feature_string)
        slide_mosaic_paths = create_slide_mosaic_mp(
                                    config=config, 
                                    all_data=all_data,
                                    method=mosaic_method, 
                                    percentile=mosaic_percentile, 
                                    mosaics_base=mosaics_base,
                                    features_folder_path=features_folder_path, 
                                    patch_size=combination_dict['tile_px'], 
                                )
        logging.info("Mosaic creation completed.")

        # ---- Patch visualization (multi-page PDF) ----
        logging.info("Creating mosaic patch visualizations...")

        if "features" in mosaic_method:
            pdf_base = os.path.join(vis_base, f"mosaics_{mosaic_method}_{tile_string}_{feature_string}")
        else:
            pdf_base = os.path.join(vis_base, f"mosaics_{mosaic_method}_{tile_string}")

        if config['experiment']['report']:
            try:
                generate_extensive_patch_selection_report_pdf(
                    config=config,
                    all_data=all_data,
                    slide_mosaic_paths=slide_mosaic_paths,
                    mosaic_method=mosaic_method,
                    pdf_base=pdf_base,
                    patch_size=combination_dict['tile_px'],
                )
            except Exception as e:
                logging.warning(f"Patch visualization failed for {mosaic_method}_{feature_string}: {e}")

        logging.info("Mosaic patch visualizations saved to PDF.")

        # ---- Generate UMAP plots (if requested) ----
        umap_base = os.path.join(vis_base, f"umap_{mosaic_method}_{feature_string}")

        if any(viz.startswith("UMAP") for viz in config.get("visualization", [])):
            run_umap_visualizations(
                config=config, 
                slide_mosaic_paths=slide_mosaic_paths, 
                mosaic_method=mosaic_method, 
                output_base=umap_base
            )

        # ---- Similar Image Search Benchmark ----
        search_string = combination_dict.get('search_method', 'yottixel-10')

        # Parse search method and k
        if '-' in search_string:
            search_method, k = search_string.split('-')
            k = int(k)
        else:
            search_method = search_string
            k = 5

        logging.info(f"Running leave-one-patient-out evaluation using {search_method} retrieving {k} slides...")
        # Run search method
        if search_method.lower() == 'yottixel':
            search_database = YottixelDatabase(config=config, slide_mosaic_paths=slide_mosaic_paths, k=k)
            results = search_database.leave_one_patient_out()
        elif search_method.lower() == 'sish':
            search_database = SISHDatabase(config=config, slide_mosaic_paths=slide_mosaic_paths, k=k, mosaic_string=f"{mosaic_method}_{feature_string}")
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
                generate_image_retrieval_report_by_class(
                    results=results,
                    all_data=all_data,
                    output_path=combo_eval_folder
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

###########################################################################################################
###########################################################################################################
"""def compute_hit_at_k(results, k):
    
    hit_counts = defaultdict(int)  # Count of successful hits per label
    total_counts = defaultdict(int)  # Total number of queries per label

    for result in results:
        query_label = result['query_label']
        if len(result['top_k']) < k:
            continue  # Skip if not enough results

        # Get top-k retrieved labels
        top_k_labels = [slide['label'] for slide in result['top_k'][:k]]

        total_counts[query_label] += 1  # Track number of queries for this label
        if query_label in top_k_labels:
            hit_counts[query_label] += 1  # Count as a hit if label is present in top-k

    # Compute hit rate per label
    hit_at_k = {label: hit_counts[label] / total_counts[label] for label in total_counts}
    return hit_at_k


def compute_mmv_at_k(results, k):
    
    correct_counts = defaultdict(int)  # Count of correct majority votes per label
    total_counts = defaultdict(int)  # Total number of queries per label

    for result in results:
        query_label = result['query_label']
        if len(result['top_k']) < k:
            continue  # Skip if not enough results

        # Get top-k retrieved labels
        top_k_labels = [slide['label'] for slide in result['top_k'][:k]]

        total_counts[query_label] += 1
        most_common_label = Counter(top_k_labels).most_common(1)[0][0]  # Most frequent label
        if most_common_label == query_label:
            correct_counts[query_label] += 1  # Count as correct if label matches query

    # Compute majority vote accuracy per label
    mmv_at_k = {label: correct_counts[label] / total_counts[label] for label in total_counts}
    return mmv_at_k

def compute_map_at_k(results, k):
    
    ap_per_label = defaultdict(list)  # Stores average precision values per label

    for result in results:
        query_label = result['query_label']
        if len(result['top_k']) < k:
            continue  # Skip if not enough results

        # Get top-k retrieved labels
        top_k_labels = [slide['label'] for slide in result['top_k'][:k]]

        num_relevant = 0  # Count of correct matches so far
        precisions = []  # Store precision at each relevant position
        for i, label in enumerate(top_k_labels):
            if label == query_label:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))  # Precision@i+1

        ap = np.mean(precisions) if precisions else 0.0  # Average precision for this query
        ap_per_label[query_label].append(ap)

    # Compute mean AP per label
    map_at_k = {label: np.mean(aps) for label, aps in ap_per_label.items()}
    return map_at_k

def evaluate_retrieval_metrics(
    results: List[Dict],
    metric_names: List[str],
    count_short_as_miss: bool = True
) -> Dict:
    
    parsed = parse_metric_names(metric_names)  # Parse metric types and k-values from strings
    metrics = {}

    print(parsed)  # Likely for debugging — consider removing or logging if finalized

    for mtype, ks in parsed.items():
        for k in ks:
            # Filter out queries with fewer than k results
            valid = [r for r in results if len(r.get('top_k', [])) >= k]
            if not valid:
                logging.warning(f"No valid queries for {mtype}@{k}")
                continue

            # Compute per-class scores
            if mtype == 'hit':
                per_class = compute_hit_at_k(valid, k)
            elif mtype == 'mmv':
                per_class = compute_mmv_at_k(valid, k)
            elif mtype == 'map':
                per_class = compute_map_at_k(valid, k)
            else:
                logging.warning(f"Unknown metric type: {mtype}")
                continue

            key = f"{mtype}_at_{k}"
            metrics[key] = per_class

            # Compute macro average (mean of per-class values)
            macro = np.mean(list(per_class.values())) if per_class else 0.0
            metrics[f"{key}_macro"] = {'all': float(macro)}

            # Compute micro average (based on all valid queries)
            if mtype == 'hit':
                total_q = len(valid)
                hits = sum(
                    1 for r in valid
                    if r['query_label'] in [s['label'] for s in r['top_k'][:k]]
                )
                micro = hits / total_q if total_q else 0.0

            elif mtype == 'mmv':
                from collections import Counter
                total_q = len(valid)
                correct = 0
                for r in valid:
                    labels = [s['label'] for s in r['top_k'][:k]]
                    if Counter(labels).most_common(1)[0][0] == r['query_label']:
                        correct += 1
                micro = correct / total_q if total_q else 0.0

            elif mtype == 'map':
                ap_list = []
                for r in valid:
                    labels = [s['label'] for s in r['top_k'][:k]]
                    num_rel = 0
                    precisions = []
                    for idx, lab in enumerate(labels):
                        if lab == r['query_label']:
                            num_rel += 1
                            precisions.append(num_rel / (idx + 1))
                    ap_list.append(np.mean(precisions) if precisions else 0.0)
                micro = float(np.mean(ap_list)) if ap_list else 0.0

            else:
                logging.warning(f"Unknown metric type for micro average: {mtype}")
                continue

            metrics[f"{key}_micro"] = {'all': micro}

    return metrics"""

"""def create_slide_mosaic(config, all_data, method, percentile, string_without_mil):

    logging.info("Running mosaic creation using method: %s", method)

    # Dynamically load the patch selection function from the patch_selection module
    method_func_name = f"{method.lower()}_patch_selection"
    try:
        patch_selection_module = importlib.import_module("pathbench.image_retrieval.patch_selection")
        patch_selection_fn = getattr(patch_selection_module, method_func_name)
    except (ImportError, AttributeError):
        raise ValueError(f"Patch selection function '{method_func_name}' not found in patch_selection module.")

    outdir = os.path.join("experiments", config['experiment']['project_name'], "bags")
    os.makedirs(outdir, exist_ok=True)
    bags_dir = os.path.join(outdir, string_without_mil)

    for tfrecord_path in tqdm(all_data.tfrecords(), desc="Processing slides"):
        base_path, _ = os.path.splitext(tfrecord_path)
        slide_id = sf.TFRecord(tfrecord_path)[0].get("slide", os.path.basename(tfrecord_path))

        # Define paths
        npz_output_path = f"{base_path}_{method}.index.npz"
        npz_input_path = f"{base_path}.index.npz"
        features_path = os.path.join(bags_dir, f"{slide_id}.pt")
        features_output_path = os.path.join(bags_dir, f"{slide_id}_{method}.pt")

        # Skip if output features already exist
        if os.path.exists(features_output_path):
            continue

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found for slide {slide_id} at {features_path}")
        
        features = torch.load(features_path)

        # ---- Step 1: Check for existing patch selection ----
        if os.path.exists(npz_output_path):
            mosaic_data = np.load(npz_output_path)
            if "selected_indices" not in mosaic_data:
                logging.warning(f"Missing selected_indices in {npz_output_path}, recomputing mosaic.")
                recompute_mosaic = True
            else:
                selected_indices = mosaic_data["selected_indices"]
                recompute_mosaic = False
        else:
            recompute_mosaic = True

        # ---- Step 2: Run mosaic selection method if needed ----
        if recompute_mosaic:
            logging.info(f"Running {method} patch selection for slide {slide_id}...")

            tfr_temp = sf.TFRecord(tfrecord_path)
            patches, _ = load_patches_from_tfr(tfr_temp)

            # Call the dynamically selected patch selection function
            selected_indices = patch_selection_fn(patches, coordinates, features, percentile)

            # Save filtered index with selected indices
            save_selected_patches_npz(npz_input_path, selected_indices, npz_output_path)

        # ---- Step 3: Save selected features ----
        selected_features = features[selected_indices]
        torch.save(selected_features, features_output_path)

        logging.info(f"Saved {len(selected_indices)} features for slide {slide_id}")"""

"""def save_selected_patches_npz(original_npz_path, selected_indices, output_npz_path):

    logging.info(f"Saving filtered NPZ index from {original_npz_path} to {output_npz_path}...")

    # Load original index
    data = np.load(original_npz_path)
    new_data = {}

    # Filter all existing fields
    for key in data.files:
        new_data[key] = data[key][selected_indices]

    # Save the selected indices for later reference (important!)
    new_data["selected_indices"] = np.array(selected_indices)

    # Save to new .npz file
    np.savez(output_npz_path, **new_data)
    logging.info(f"Filtered NPZ index with selected_indices saved to {output_npz_path}.")"""

"""def visualize_selected_patches_slide(
    all_data,
    slide_id: str,
    tfr_path: str,
    npz_path: str,
    method: str,
    patch_size: int = 256,
    thumb_size: int = 1024,
    save_path: str = None
):

    npz = np.load(npz_path)
    locations = npz["locations"]
    selected_indices = npz["selected_indices"]

    tfr = sf.TFRecord(tfr_path)
    patches = [np.array(sf.io.decode_image(bytes(tfr[idx]['image_raw']))) for idx in selected_indices]

    slide_path = all_data.find_slide(slide=slide_id)
    if slide_path is None:
        logging.info(f"Slide path not found for slide ID: {slide_id}")
        return None

    slide = openslide.OpenSlide(slide_path)
    thumb = slide.get_thumbnail((thumb_size, thumb_size))
    dims = slide.dimensions
    scale_x = thumb.width / dims[0]
    scale_y = thumb.height / dims[1]
    scaled_locations = locations * np.array([scale_x, scale_y])

    num_patches = len(patches)
    num_cols = min(6, num_patches)
    num_rows = math.ceil(num_patches / num_cols)

    fig = plt.figure(figsize=(12, 6 + 2 * num_rows))
    gs = GridSpec(2 + num_rows, num_cols, figure=fig, height_ratios=[4, 0.2] + [1]*num_rows)

    ax_thumb = fig.add_subplot(gs[0, :])
    ax_thumb.imshow(thumb)
    ax_thumb.set_title(f"{slide_id} - Selected patches ({method})")
    ax_thumb.axis("off")
    for i, (x, y) in enumerate(scaled_locations):
        rect = Rectangle((x, y), patch_size * scale_x, patch_size * scale_y,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax_thumb.add_patch(rect)
        ax_thumb.text(x, y, str(i + 1), color='white', fontsize=8, backgroundcolor='red')

    for i, patch in enumerate(patches):
        row = 2 + i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(patch)
        ax.axis("off")
        ax.set_title(f"{i + 1}", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        loggin.info(f"Saved patch visualization to: {save_path}")
        return None

    return fig
"""

"""def generate_patch_selection_report_pdf(config, all_data, mosaic_method, string_without_mil):
    
    vis_dir = os.path.join("experiments", config["experiment"]["project_name"], "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    pdf_path = os.path.join(vis_dir, f"patches_{mosaic_method}_{string_without_mil}.pdf")

    bags_dir = os.path.join("experiments", config["experiment"]["project_name"], "bags", string_without_mil)

    with PdfPages(pdf_path) as pdf:
        for tfrecord_path in all_data.tfrecords():
            base_path, _ = os.path.splitext(tfrecord_path)
            slide_id = sf.TFRecord(tfrecord_path)[0].get("slide", os.path.basename(tfrecord_path))

            npz_path = f"{base_path}_{mosaic_method}.index.npz"
            if not os.path.exists(npz_path):
                continue

            try:
                fig = visualize_selected_patches_slide(
                    all_data=all_data,
                    slide_id=slide_id,
                    tfr_path=tfrecord_path,
                    npz_path=npz_path,
                    method=mosaic_method,
                    save_path=None
                )
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
            except Exception as e:
                logging.info(f"Skipping {slide_id} due to error: {e}")

    logging.info(f"Saved multi-page PDF of patch visualizations to: {pdf_path}")
"""

"""def plot_slide_umap(config, all_data, mosaic_method, aggregation_method, save_string, umap_params):

    from collections.abc import Iterable
    logging.info(f"Generating UMAP with method={mosaic_method}, aggregation={aggregation_method}, params={umap_params}")

    # Load annotations
    annotations = pd.read_csv(config['experiment']['annotation_file'])
    annotations = annotations.set_index("slide")

    # Prepare features and labels
    slide_features = []
    slide_labels = []

    out_dir = os.path.join("experiments", config['experiment']['project_name'], "bags")
    bags_dir = os.path.join(out_dir, save_string)

    for tfrecord_path in tqdm(all_data.tfrecords(), desc="Aggregating slide features"):
        tfr_temp = sf.TFRecord(tfrecord_path)
        slide_id = tfr_temp[0].get("slide", os.path.basename(tfrecord_path))

        mosaic_path = os.path.join(bags_dir, f"{slide_id}_{mosaic_method}.pt")
        if not os.path.exists(mosaic_path):
            logging.warning(f"Skipping slide {slide_id}: mosaic features not found.")
            continue

        try:
            label = annotations.loc[slide_id]["category"]
        except KeyError:
            logging.warning(f"Slide {slide_id} not found in annotation file.")
            continue

        features = torch.load(mosaic_path).numpy()

        # Aggregate if specified
        if aggregation_method is None or aggregation_method.lower() == "none":
            slide_features.extend(features)
            slide_labels.extend([label] * features.shape[0])
        elif aggregation_method == "mean":
            slide_features.append(features.mean(axis=0))
            slide_labels.append(label)
        elif aggregation_method == "median":
            slide_features.append(np.median(features, axis=0))
            slide_labels.append(label)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    if not slide_features:
        logging.error("No valid features found for UMAP.")
        return

    slide_features = np.array(slide_features)

    # Run UMAP
    reducer = umap.UMAP(
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        metric=umap_params.get("metric", "euclidean"),
        random_state=1
    )
    embedding = reducer.fit_transform(slide_features)

    # Plot UMAP
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=slide_labels, palette="tab10", s=80, alpha=0.8)
    plt.title(f"UMAP ({aggregation_method}) using {mosaic_method}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save plot
    save_dir = os.path.join("experiments", config['experiment']['project_name'], "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"umap_{mosaic_method}_{aggregation_method}_{save_string}.png")

    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved UMAP plot to {save_path}")
"""

"""def run_umap_visualizations(config, all_data, mosaic_method, save_string):
    
    visualizations = config.get("visualization", [])
    umap_params_raw = config.get("umap_parameters", [])
    umap_params = {list(d.keys())[0]: list(d.values())[0] for d in umap_params_raw}

    for viz in visualizations:
        if not viz.startswith("UMAP"):
            continue

        # Get aggregation method, default to "mean" if not specified
        _, aggregation_method = viz.split("-") if "-" in viz else ("UMAP", "mean")

        if aggregation_method.lower() == "none":
            aggregation_method = None

        try:
            logging.info(f"Generating UMAP ({aggregation_method}) for current combination...")
            plot_slide_umap(
                config=config,
                all_data=all_data,
                mosaic_method=mosaic_method,
                aggregation_method=aggregation_method,
                save_string=save_string,
                umap_params=umap_params
            )
        except Exception as e:
            logging.warning(f"UMAP generation failed for {save_string} ({aggregation_method}): {e}")
"""

"""def save_patches_json(patches, path):

    serializable_patches = []
    for patch in patches:
        serializable_patch = {
            'loc': patch['loc'],
            'wsi_loc': patch['wsi_loc'],
            'feature': patch['feature'].tolist() if isinstance(patch['feature'], np.ndarray) else list(patch['feature']),
            'raw_patch': base64.b64encode(patch['raw_patch']).decode('utf-8'),
            'rgb_histogram': patch['rgb_histogram']
        }
        serializable_patches.append(serializable_patch)

    with open(path, 'w') as f:
        json.dump(serializable_patches, f, indent=2)

def load_patches_json(path):
    
    try:
        with open(path, 'r') as f:
            patch_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to load JSON from {path}. File may be empty or corrupted: {e}")

    patches = []
    for item in patch_data:
        patch = {
            'loc': item['loc'],
            'wsi_loc': item['wsi_loc'],
            'feature': np.array(item['feature'], dtype=np.float32),
            'raw_patch': base64.b64decode(item['raw_patch']),
            'rgb_histogram': item['rgb_histogram']
        }
        patches.append(patch)

    return patches"""

"""def generate_image_retrieval_report_pdf(results, all_data, pdf_path):

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for result in results:
            try:
                fig = visualize_retrieval_result(result, all_data)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                logging.info(f"Visualization failed for {result['query_slide_id']}: {e}")

    logging.info(f"Saved retrieval report PDF to: {pdf_path}")"""

"""def load_patch_dicts_from_tfr(tfr_temp, features_npz, features_tensor, patch_size):
    
    patches = []
    feature_locs = features_npz['arr_0']
    features_array = features_tensor.numpy() if hasattr(features_tensor, 'numpy') else features_tensor

    for tfr_record in tfr_temp:
        # Decode patch image from raw bytes
        img = sf.io.decode_image(bytes(tfr_record['image_raw']))
        img_np = np.array(img)

        # Extract patch location from record
        location = np.array([int(tfr_record['loc_x']), int(tfr_record['loc_y'])])

        # Match location with corresponding feature index
        idx = np.where((feature_locs == location).all(axis=1))[0]
        if len(idx) == 0:
            continue  # Skip if no match is found
        idx = idx[0]

        # Compute normalized mean RGB value (used as simple histogram)
        rgb_hist = img_np.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.

        # Store all patch-related information
        patch_info = {
            'loc': location.tolist(),
            'wsi_loc': (location // patch_size).tolist(),
            'feature': features_array[idx],
            'raw_patch': tfr_record['image_raw'],  # Keep raw patch as bytes for size efficiency
            'rgb_histogram': rgb_hist
        }

        patches.append(patch_info)

    return patches


def save_patch_dicts_pickle(patch_dicts, path, features_tensor, compress=3):
    
    # Ensure features_tensor is a NumPy array (if not already)
    if hasattr(features_tensor, 'numpy'):
        features_tensor = features_tensor.numpy()

    simplified = []

    for i, patch in enumerate(patch_dicts):
        patch_copy = patch.copy()
        
        # Replace feature with index to reduce file size
        if 'feature' in patch_copy:
            patch_copy.pop('feature')
            patch_copy['feature_index'] = i  # Index matches the order of the original feature tensor

        simplified.append(patch_copy)

    # Save to disk using joblib with specified compression
    joblib.dump(simplified, path, compress=compress)

def load_patch_dicts_pickle(path, features_tensor):

    # Ensure features_tensor is in NumPy format for indexing
    if hasattr(features_tensor, 'numpy'):
        features_tensor = features_tensor.numpy()

    # Load patch dicts with 'feature_index' from disk
    patch_dicts = joblib.load(path)
    reconstructed = []

    for patch in patch_dicts:
        patch_copy = patch.copy()

        # Replace 'feature_index' with actual feature vector
        if 'feature_index' in patch_copy:
            idx = patch_copy.pop('feature_index')
            patch_copy['feature'] = features_tensor[idx]

        reconstructed.append(patch_copy)

    return reconstructed"""

"""def generate_patch_selection_report_pdf(config, all_data, slide_mosaics, mosaic_method, pdf_path, patch_size=None):

# if the caller didn’t supply patch_size, grab it from config:
if patch_size is None:
    px_list = config['benchmark_parameters']['tile_px']
    # fall back to first entry if it’s a list
    patch_size = px_list[0] if isinstance(px_list, (list, tuple)) else px_list

with PdfPages(pdf_path) as pdf:
    for slide_id, mosaic_patches in slide_mosaics.items():
        slide_path = all_data.find_slide(slide=slide_id)
        if slide_path is None:
            logging.info(f"Slide path not found for slide ID: {slide_id}")
            continue

        fig = visualize_selected_patches_slide(
            slide_id=slide_id,
            slide_path=slide_path,
            mosaic_patches=mosaic_patches,
            method=mosaic_method,
            patch_size=patch_size
        )
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)
logging.info(f"Saved multi-page PDF of patch visualizations to: {pdf_path}")"""