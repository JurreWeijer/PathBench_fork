import logging 
import numpy as np
from sklearn.cluster import KMeans

from .base import MosaicSelector
from .registry import register_mosaic_selectors

logger = logging.getLogger(__name__)

@register_mosaic_selectors
class HSHRFeatures(MosaicSelector):
    """
    HSHR-Features Mosaic Selection
    -------------------------------
    Clusters patch *features* with k-means (k = desired #patches),
    then selects, from each cluster, the single patch nearest to the
    cluster centroid.

    Args:
        patches (list of dict): each with:
          - 'feature' : 1D embedding (array-like)
          - 'loc'     : (x, y) integer coordinates (optional but shown in return)
        n_patches (int): number of patches to select (k in k-means)

    Returns:
        selected_indices (list[int]): indices into `patches` of chosen patches
        group_ids        (np.ndarray[int]): compact 0..G-1 label per patch
        coords           (np.ndarray[int, shape (N,2)]): all (x,y) coords
        groups           (dict[int, np.ndarray]): cluster_id -> member indices
    """
    name = "HSHR_features"
    HYPERPARAMS = {
        "n_patches": {
            "type": int, "default": 25, "min": 1,
            "help": "Desired number of patches to select (k in k-means).",
            "attr": "n_patches",
            "include_in_id": True, "id_order": 0,
        },
    }

    def __init__(self, params, config):
        super().__init__(params, config)
        self.n_patches = self._get_hp("n_patches")
        self.random_state = (self.config.get("experiment", {}) or {}).get("random_state", None)

    def run(self, patches, **_):
        if len(patches) == 0:
            logger.warning("HSHRFeatures: empty patch list.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        # Build feature matrix and coords
        try:
            features = np.asarray([p["feature"] for p in patches], dtype=float)
        except Exception as e:
            raise ValueError(f"HSHRFeatures: could not read 'feature' from patches ({e}).")
        coords = np.array(
            [[int(p.get("loc", (0, 0))[0]), int(p.get("loc", (0, 0))[1])] for p in patches],
            dtype=int
        )

        # k must be <= #samples and >= 1
        k = int(self.n_patches) if self.n_patches is not None else 1
        k = max(1, min(k, len(patches)))

        # Cluster in feature space
        km = KMeans(
            n_clusters=k,
            n_init="auto",
            random_state=self.config["experiment"].get("random_state", None)
        )
        labels = km.fit_predict(features)               # shape (N,)
        centers = km.cluster_centers_                   # shape (k, D)

        # Map original labels to compact 0..G-1 just in case
        unique_bins, group_ids = np.unique(labels, return_inverse=True)

        # Build groups: cluster_id -> np.ndarray of member indices
        groups = {g: np.where(group_ids == g)[0] for g in range(len(unique_bins))}

        # For each cluster, pick member closest to its centroid
        selected = []
        for g, member_idx in groups.items():
            if member_idx.size == 0:
                continue
            feats_g = features[member_idx]              # (M, D)
            center_g = centers[g]                       # (D,)
            # squared euclidean distances
            d2 = np.sum((feats_g - center_g) ** 2, axis=1)
            # choose the global index of the closest member
            pick_local = int(np.argmin(d2))
            selected.append(int(member_idx[pick_local]))

        return selected, group_ids.astype(int), coords, groups