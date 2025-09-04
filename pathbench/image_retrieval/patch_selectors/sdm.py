import logging 
import numpy as np
from sklearn.cluster import KMeans

from .base import PatchSelector
from .registry import register_patch_selectors

@register_patch_selectors
class SDMFeatures(PatchSelector):
    name = "SDM_features"

    """
    Selection of Distinct Morphologies (SDM)
    ----------------------------------------
    Unsupervisedly selects one patch per “distance bin” from the centroid of all
    patch embeddings. By sampling uniformly across these bins, SDM ensures a mosaic
    that captures the full spectrum of morphological variation in the slide.

    Computes the Euclidean distance of each patch’s feature vector to the global
    centroid, discretizes those distances into integer “bins,” and then picks one
    representative patch from each bin via a reproducible random choice.

    Args:
        patches (list of dict):
            List of patch dictionaries, each containing:
              - 'feature': 1D numpy array embedding of the patch.
              - any other metadata (ignored here).
        percentage_selected (float, optional):
            Unused—present for interface consistency with other selectors.
        random_state (int):
            Seed for the random number generator to ensure reproducibility.

    Returns:
        selected (list[int]): one chosen index per group.
        group_ids (np.ndarray[int]): compact 0..G-1 label per patch (len == len(patches)).
        coords (np.ndarray[int]): Nx2 array of [x, y] per patch.
        groups (dict[int, np.ndarray]): group_id -> array of member indices.
    """
    
    def run(self, patches, **_):

        if not patches:
            logging.warning("Empty patch list provided to SDM.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        # ---- Stack features and compute centroid ----
        feats = np.stack([p['feature'] for p in patches], axis=0).astype(float)
        if not np.all(np.isfinite(feats)):
            raise ValueError("Non-finite values in features.")

        centroid = feats.mean(axis=0)

        # ---- Distances & raw bins (keep default behavior) ----
        dists = np.linalg.norm(feats - centroid[None, :], axis=1)
        raw_bin_ids = np.rint(dists).astype(int)

        # ---- Make bins compact: 0..G-1 in order of first appearance ----
        unique_bins, group_ids = np.unique(raw_bin_ids, return_inverse=True)

        # ---- Build groups (indices per group) ----
        groups = {g: np.where(group_ids == g)[0] for g in range(len(unique_bins))}

        # ---- Reproducible selection: one index per group ----
        rng = np.random.default_rng(self.config.get("experiment", {}).get("random_state", None))
        selected = [int(rng.choice(idx_arr)) for idx_arr in groups.values()]

        # ---- Coords alongside groups ----
        coords = np.array([[int(p['loc'][0]), int(p['loc'][1])] for p in patches], dtype=int)

        return selected, group_ids, coords, groups