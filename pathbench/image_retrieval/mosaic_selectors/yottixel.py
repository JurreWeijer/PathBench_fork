# ------------------------------------------------------------------------------
# Yottixel Mosaic Selection:
#   Based on code from the official Yottixel repository:
#     https://github.com/KimiaLabMayo/yottixel
#   Source Paper:
#     Kalra, S., Tizhoosh, H.R., Choi, C., et al.
#     “Yottixel – An Image Search Engine for Large Archives of Histopathology Whole Slide Images.”
#     Medical Image Analysis 65 (2020): 101757. https://doi.org/10.1016/j.media.2020.101757
#
# RetCCL-inspired Yottixel variant:
#   Source Paper:
#     Wang, X., Du, Y., Yang, S., et al.
#     “RetCCL: Clustering-Guided Contrastive Learning for Whole-Slide Image Retrieval.”
#     Medical Image Analysis 83 (2023): 102645. https://doi.org/10.1016/j.media.2022.102645
# ------------------------------------------------------------------------------

import logging 
import numpy as np
from sklearn.cluster import KMeans

from .base import MosaicSelector
from .registry import register_mosaic_selectors

@register_mosaic_selectors
class YottixelRGB(MosaicSelector):
    name = "Yottixel_rgb"
    param_key = "percentage_selected"

    """
    Yottixel RGB Mosaic Selection
    ----------------------------
    Implements the Yottixel mosaic patch selection strategy based on RGB histogram
    clustering and spatial clustering. The algorithm identifies representative patches
    across tissue regions with varying color characteristics.

    The method clusters patches based on their color histograms using k-means, then
    within each color cluster, it performs a second clustering based on spatial location.
    A fixed percentage of spatially diverse representatives is selected from each cluster.

    Args:
        patches (list of dict): each with:
        - 'rgb_histogram': normalized RGB histogram (1D array)
        - 'loc': (x, y) slide coordinates
        percentage_selected (float):  Percentage of patches to select from each color cluster.

    Returns:
        list: Indices of selected patches from the input list.
    
    Reference:
        Kalra, Shivam, H.R. Tizhoosh, Charles Choi, Sultaan Shah, Phedias Diamandis, Clinton J.V. Campbell, and Liron Pantanowitz. 
        “Yottixel - An Image Search Engine for Large Archives of Histopathology Whole Slide Images.” Medical Image Analysis 65 
        (October 2020): 101757. https://doi.org/10.1016/j.media.2020.101757.
    """

    def run(self, patches, *, percentage_selected: int, **_):	
        
        kmeans_clusters = 9  # TODO: move to config if needed

        if len(patches) == 0:
            logging.warning("Empty patch list provided to Yottixel selection.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        # ---- Stage 1: Color clustering ----
        rgb_hist = np.array([p['rgb_histogram'] for p in patches])
        kmeans_clusters = min(kmeans_clusters, len(patches))  # Cap clusters to number of patches
        kmeans_color = KMeans(
            n_clusters=kmeans_clusters,
            random_state=self.config["experiment"].get("random_state", None)
        )
        color_labels_raw = kmeans_color.fit_predict(rgb_hist)

        # Compact labels 0..G-1 and build groups
        unique_bins, group_ids = np.unique(color_labels_raw, return_inverse=True)
        groups = {g: np.where(group_ids == g)[0] for g in range(len(unique_bins))}

        # ---- Stage 2: Spatial clustering within each color group (selection logic unchanged) ----
        selected = []
        for g in range(len(unique_bins)):
            member_idx = groups[g]
            if member_idx.size == 0:
                continue

            cluster_patches = [patches[i] for i in member_idx]
            n_select = max(1, int(len(cluster_patches) * percentage_selected / 100))

            loc_features = np.asarray([p['loc'] for p in cluster_patches], dtype=float)
            kmeans_loc = KMeans(
                n_clusters=n_select,
                random_state=self.config["experiment"].get("random_state", None)
            )
            dists = kmeans_loc.fit_transform(loc_features)

            used_local = set()
            for c in range(n_select):
                # nearest unused to center c
                sorted_local = np.argsort(dists[:, c])
                for sidx in sorted_local:
                    if sidx not in used_local:
                        used_local.add(sidx)
                        selected.append(member_idx[sidx].item())
                        break

        # ---- Coords alongside groups ----
        coords = np.array([[int(p['loc'][0]), int(p['loc'][1])] for p in patches], dtype=int)

        return selected, group_ids.astype(int), coords, groups

@register_mosaic_selectors
class YottixelFeatures(MosaicSelector):
    name = "Yottixel_features"
    param_key = "percentage_selected"

    """
    Yottixel-Features Mosaic Selection (RetCCL-Inspired)
    ---------------------------------------------------
    Implements a Yottixel-style two-stage clustering procedure using deep learning
    features instead of RGB histograms. This method balances feature diversity and
    spatial representativeness.

    Patches are first clustered in feature space using k-means. Within each feature
    cluster, spatial clustering is performed to select a percentage of representative
    patches, ensuring a mosaic that reflects both semantic and spatial variance.

    Args:
        patches (list of dict): each with:
        - 'feature': deep feature embedding (1D array)
        - 'loc': (x, y) slide coordinates
        percentage_selected (float): Percentage of patches to select from each feature cluster.

    Returns:
        list: Indices of selected patches from the input list.

    Reference:
        Wang, Xiyue, Yuexi Du, Sen Yang, Jun Zhang, Minghui Wang, Jing Zhang, Wei Yang, Junzhou Huang, and Xiao Han. 
        “RetCCL: Clustering-Guided Contrastive Learning for Whole-Slide Image Retrieval.” Medical Image Analysis 83 
        (January 1, 2023): 102645. https://doi.org/10.1016/j.media.2022.102645.
        
    """

    def run(self, patches, *, percentage_selected: int, **_):

        if len(patches) == 0:
            logging.warning("Empty patch list provided.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        kmeans_clusters = 9  # TODO: move to config if needed

        # ---- Stage 1: Feature clustering ----
        features = np.asarray([p['feature'] for p in patches], dtype=float)
        kmeans_clusters = min(kmeans_clusters, len(patches))
        kmeans_feat = KMeans(
            n_clusters=kmeans_clusters,
            random_state=self.config["experiment"].get("random_state", None)
        )
        feat_labels_raw = kmeans_feat.fit_predict(features)

        # Compact labels 0..G-1 (stable) for group tracking
        unique_bins, group_ids = np.unique(feat_labels_raw, return_inverse=True)

        # Build groups (indices per first-stage cluster)
        groups = {g: np.where(group_ids == g)[0] for g in range(len(unique_bins))}

        # ---- Selection logic (unchanged): choose reps per feature-cluster ----
        selected = []
        for g in range(len(unique_bins)):
            member_idx = groups[g]
            if member_idx.size == 0:
                continue

            # Gather cluster-local patches in original order
            cluster_patches = [patches[i] for i in member_idx]
            n_select = max(1, int(len(cluster_patches) * percentage_selected / 100))

            if n_select == 1:
                # pick the first member (same behavior as before)
                selected.append(member_idx[0].item())
                continue

            # ---- Stage 2: Spatial clustering ----
            locs = np.asarray([p['loc'] for p in cluster_patches], dtype=float)
            kmeans_loc = KMeans(
                n_clusters=n_select,
                random_state=self.config["experiment"].get("random_state", None)
            )
            dists = kmeans_loc.fit_transform(locs)

            used_local = set()
            for c in range(n_select):
                # nearest unused to center c
                sorted_local = np.argsort(dists[:, c])
                for sidx in sorted_local:
                    if sidx not in used_local:
                        used_local.add(sidx)
                        # map cluster-local index back to global index
                        selected.append(member_idx[sidx].item())
                        break

        # ---- Coords alongside groups ----
        coords = np.array([[int(p['loc'][0]), int(p['loc'][1])] for p in patches], dtype=int)

        return selected, group_ids.astype(int), coords, groups