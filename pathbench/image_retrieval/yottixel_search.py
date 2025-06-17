# ------------------------------------------------------------------------------
# BoB and YottixelDatabase Classes
# Source: Adapted from the official Yottixel implementation
# GitHub: https://github.com/KimiaLabMayo/yottixel/blob/main/yottixel_kimianet/helper_functions.py
# Reference: 
#   Kalra, S., Tizhoosh, H.R., Choi, C., et al. 
#   “Yottixel – An Image Search Engine for Large Archives of Histopathology Whole Slide Images.” 
#   Medical Image Analysis 65 (2020): 101757. https://doi.org/10.1016/j.media.2020.101757
# ------------------------------------------------------------------------------

import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
from bitarray import bitarray
import bitarray.util as butil
import torch
from ..image_retrieval.utils import load_patch_dicts_pickle
import logging

logger = logging.getLogger(__name__)

class BoB:
    """
    Bag of Barcodes (BoB) representation for a slide.

    Converts each feature vector into a binary barcode using bitarrays,
    and provides a method to compute distances to other BoBs using median
    of minimum XOR distances.

    Args:
        barcodes (np.ndarray): Binary (0/1) barcode matrix derived from features.
        slide_id (str): ID of the slide this BoB represents.
        patient_id (str): Patient ID for exclusion in evaluation.
        label (str): Ground-truth label (e.g., tumor type).
    """
    def __init__(self, barcodes, slide_id, patient_id, label):
        self.slide_id = slide_id
        self.patient_id = patient_id
        self.label = label

        # Convert binary numpy arrays to bitarrays for efficient XOR operations
        self.barcodes = [bitarray(b.tolist()) for b in barcodes]

    def distance(self, other_bob):
        """
        Compute the distance to another BoB using minimum XOR distance per barcode.

        Args:
            other_bob (BoB): Another BoB instance.

        Returns:
            float: Median of minimum XOR distances across all barcodes.
        """
        total_dist = []
        for feat in self.barcodes:
            # Compute XOR distance between this feature and all in other BoB
            distances = [butil.count_xor(feat, b) for b in other_bob.barcodes]
            total_dist.append(np.min(distances))
        return np.median(total_dist)

class YottixelDatabase:
    """
    A simple database of BoBs (slides) supporting image retrieval.

    Supports leave-one-patient-out benchmarking using BoB-based search and
    majority vote among top-k most similar slides.

    Args:
        config (dict): Configuration containing experiment metadata.
        slide_mosaics (dict): Mapping from slide ID to list of patch dictionaries.
        k (int): Number of nearest neighbors to use in retrieval.
    """
    def __init__(self, config: dict, slide_mosaic_paths: dict, k: int = 3):
        self.k = k
        self.bobs = []

        # Load slide annotations
        annotation_path = config['experiment']['annotation_file']
        annotations = pd.read_csv(annotation_path).set_index("slide")

        # Create a BoB for each slide using binarized features
        for slide_id, mosaic_pkl in tqdm(slide_mosaic_paths.items(), desc="Building BoBs"):
            if slide_id not in annotations.index:
                continue

            label = annotations.loc[slide_id]["category"]
            patient_id = annotations.loc[slide_id]["patient"]

            # ---- re-load only the selected patches (this reinserts .feature) ----
            mosaic_data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=True)
            mosaic_patches = mosaic_data["patches"]

            # 3) stack into an array shape (n_patches, feat_dim)
            patch_features = np.stack([p['feature'] for p in mosaic_patches], axis=0)

            # 4) turn each feature into a binary barcode (your chosen scheme)
            barcodes = (np.diff(patch_features, axis=1) < 0).astype(int)

            # 5) build our BoB
            bob = BoB(barcodes, slide_id, patient_id, label)
            self.bobs.append(bob)

    def predict_slide(self, query_bob: BoB) -> dict:
        """
        Predict the label of a query slide using BoB-based nearest neighbor search.

        Args:
            query_bob (BoB): BoB instance for the query slide.

        Returns:
            dict: Retrieval result with top-k matches and predicted label.
        """
        # Exclude BoBs from the same patient as the query
        atlas_bobs = [b for b in self.bobs if b.patient_id != query_bob.patient_id]
        atlas_labels = [b.label for b in atlas_bobs]
        atlas_ids = [b.slide_id for b in atlas_bobs]

        assert len(atlas_bobs) == len(atlas_labels), 'atlas_bobs and atlas_labels should have the same number of rows'

        # Compute distances and select top-k closest BoBs
        distances = np.array([query_bob.distance(bob) for bob in atlas_bobs])
        order = np.argsort(distances)[: self.k ]

        top_k_info = [
            {"slide_id": atlas_ids[i], "label": atlas_labels[i], "distance": float(distances[i])}
            for i in order
        ]

        # Perform majority voting on top-k labels
        top_k_labels = [info["label"] for info in top_k_info]
        majority_vote = Counter(top_k_labels).most_common(1)[0][0]

        return {
            "query_slide_id": query_bob.slide_id,
            "query_label": query_bob.label,
            "predicted_label": majority_vote,
            "top_k": top_k_info
        }

    def leave_one_patient_out(self):
        """
        Perform leave-one-patient-out retrieval benchmark.

        Returns:
            list: Retrieval results for each slide in the dataset.
        """
        results = []
        for query_bob in tqdm(self.bobs, desc="Leave-one-patient-out retrieval"):
            result = self.predict_slide(query_bob)
            results.append(result)
            
        return results
    
    