from tqdm import tqdm
import logging 
import numpy as np
from sklearn.cluster import KMeans

from .base import PatchSelector
from .registry import register_patch_selectors 

logger = logging.getLogger(__name__)

@register_patch_selectors
class SPLICERGB(PatchSelector):
    name = "SPLICE_rgb"
    param_key = "percentile_threshold"

    """
    SPLICE RGB Patch Selection
    --------------------------
    Applies the SPLICE algorithm using RGB color histograms to reduce redundancy
    among selected patches. This method iteratively selects patches and excludes
    neighboring ones with similar color profiles, ensuring a diverse subset.

    The SPLICE method works in a streaming fashion, computing the Euclidean
    distance between patches and suppressing nearby redundant patches within a
    distance threshold determined by a user-defined percentile.

    Args:
        patches (list of dict): List of patch dictionaries, each containing:
            - 'rgb_histogram': Normalized RGB histogram of the patch.
        percentile_threshold (float): Percentile value (0-100) for distance suppression threshold.

    Returns:
        list: Indices of selected patches from the input list.
    
    Reference:
        Alsaafin, Areej, Peyman Nejat, Abubakr Shafique, Jibran Khan, Saghir Alfasly, Ghazal Alabtah, and H. R. Tizhoosh. 
        “SPLICE -- Streamlining Digital Pathology Image Processing.” arXiv, April 26, 2024. https://doi.org/10.48550/arXiv.2404.17704.
    """ 

    def run(self, patches, *, percentile_threshold: int, **_):
        
        if patches is None or len(patches) == 0:
            logging.warning("Empty patch list provided to SPLICE-RGB.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        if percentile_threshold is None:
            raise ValueError("percentile_threshold must be specified for SPLICE.")

        # ---- Extract RGB histogram features ----
        color_features = np.array([p['rgb_histogram'] for p in patches])
        num_patches = color_features.shape[0]

        selected = []
        excluded = np.zeros(num_patches, dtype=bool)   # selection logic unchanged
        group_ids = -1 * np.ones(num_patches, dtype=int)
        groups = {}

        for i in range(num_patches):
            if excluded[i]:
                continue

            # Seed i becomes selected -> open a new group
            group_id = len(selected)
            selected.append(i)
            group_ids[i] = group_id

            ref_feat = color_features[i]
            remaining_idx = np.where(~excluded)[0]

            # Distances from this seed to remaining
            distances = np.linalg.norm(color_features[remaining_idx] - ref_feat, axis=1)
            if distances.size == 0:
                groups[group_id] = np.array([i], dtype=int)
                continue

            # Percentile-based suppression threshold
            thresh = np.percentile(distances, percentile_threshold)

            # Suppress/assign similar patches to THIS seed's group
            members = [i]
            for j, d in zip(remaining_idx, distances):
                if j == i:
                    continue
                if d < thresh:
                    excluded[j] = True
                    group_ids[j] = group_id
                    members.append(j)

            groups[group_id] = np.array(members, dtype=int)

        # Guard: assign any unassigned leftovers to nearest seed (rare)
        if (group_ids == -1).any() and len(selected) > 0:
            seeds = np.array(selected, dtype=int)
            seed_feats = color_features[seeds]
            unassigned = np.where(group_ids == -1)[0]
            for idx in unassigned:
                d = np.linalg.norm(seed_feats - color_features[idx], axis=1)
                gid = int(np.argmin(d))  # index within seeds (also group id)
                group_ids[idx] = gid
                groups[gid] = np.concatenate([groups[gid], [idx]])

        # ---- Coords ----
        coords = np.array([[int(p['loc'][0]), int(p['loc'][1])] for p in patches], dtype=int)

        return selected, group_ids.astype(int), coords, groups

@register_patch_selectors
class SPLICEFeatures(PatchSelector):
    name = "SPLICE_features"
    param_key = "percentile_threshold"

    """
    SPLICE Features Patch Selection
    -------------------------------
    Applies the SPLICE algorithm using deep learning features instead of RGB histograms
    to identify and retain a diverse set of informative patches.

    This variant operates in the same streaming selection mode as the original SPLICE,
    but uses feature embeddings (e.g., from a neural network) to compute pairwise distances.
    A patch is only selected if it differs enough from previously selected ones.

    Args:
        patches (list of dict): each with:
        - 'feature': deep feature embedding (1D array)
        - 'loc': (x, y) slide coordinates
        percentile_threshold (float): Percentile value (0-100) for distance suppression threshold.

    Returns:
        list: Indices of selected patches from the input list.

    Reference:
        Alsaafin, Areej, Peyman Nejat, Abubakr Shafique, Jibran Khan, Saghir Alfasly, Ghazal Alabtah, and H. R. Tizhoosh. 
        “SPLICE -- Streamlining Digital Pathology Image Processing.” arXiv, April 26, 2024. https://doi.org/10.48550/arXiv.2404.17704.
    """

    def run(self, patches, *, percentile_threshold: int, **_):
        
        if patches is None or len(patches) == 0:
            logging.warning("Empty patch list provided to SPLICE.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        if percentile_threshold is None:
            raise ValueError("percentile_threshold must be specified for SPLICE.")

        # ---- Extract features ----
        features = np.array([p['feature'] for p in patches])

        num_patches = features.shape[0]
        selected = []

        # Track exclusion (unchanged selection behavior) and grouping
        excluded = np.zeros(num_patches, dtype=bool)

        # group_ids: -1 until assigned. Each seed opens a new group id = len(selected) at selection time.
        group_ids = -1 * np.ones(num_patches, dtype=int)
        groups = {}

        for i in range(num_patches):
            if excluded[i]:
                continue

            # Seed i becomes a selected representative -> open a new group
            group_id = len(selected)
            selected.append(i)
            group_ids[i] = group_id  # seed belongs to its own group

            ref_feat = features[i]
            remaining_idx = np.where(~excluded)[0]

            # Distances from this seed to remaining, compute threshold
            distances = np.linalg.norm(features[remaining_idx] - ref_feat, axis=1)
            if distances.size == 0:
                groups[group_id] = np.array([i], dtype=int)
                continue

            thresh = np.percentile(distances, percentile_threshold)

            # Suppress/assign similar patches to THIS seed's group
            members = [i]
            for j, d in zip(remaining_idx, distances):
                if j == i:
                    continue
                if d < thresh:
                    excluded[j] = True
                    group_ids[j] = group_id
                    members.append(j)

            groups[group_id] = np.array(members, dtype=int)

        # Any patch that never got suppressed by any seed but also wasn't picked (shouldn't happen
        # with this loop) would still be -1; guard (assign to nearest seed if needed). Usually none.
        if (group_ids == -1).any() and len(selected) > 0:
            seeds = np.array(selected, dtype=int)
            seed_feats = features[seeds]
            # Assign stragglers to nearest seed (doesn't change selected set)
            unassigned = np.where(group_ids == -1)[0]
            for idx in unassigned:
                d = np.linalg.norm(seed_feats - features[idx], axis=1)
                gid = int(np.argmin(d))  # index within seeds
                group_ids[idx] = gid
                groups[gid] = np.concatenate([groups[gid], [idx]])

        # ---- Coords ----
        coords = np.array([[int(p['loc'][0]), int(p['loc'][1])] for p in patches], dtype=int)

        return selected, group_ids.astype(int), coords, groups