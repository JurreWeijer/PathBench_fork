# ------------------------------------------------------------------------------
# Patch Selection Methods (SPLICE, Yottixel Variants, and SDM)
#
# SPLICE (Streaming Patch Selection):
#   Source Paper:
#     Alsaafin, A., Nejat, P., Shafique, A., et al. 
#     “SPLICE -- Streamlining Digital Pathology Image Processing.” arXiv, 2024.
#     https://doi.org/10.48550/arXiv.2404.17704
#   No official GitHub available.
#
# Yottixel Patch Selection:
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
#
# SDM (Selection of Distinct Morphologies):
#   Source Paper:
#     Shafique, A., Fröhlich, K., Alsaafin, A., et al.
#     “Selection of Distinct Morphologies to Divide & Conquer Gigapixel Pathology Images.”
#     Medical Image Analysis (2023). DOI:10.1016/j.media.2023.102123
#   No official code release; reference implementation provided in this script.
# ------------------------------------------------------------------------------

from tqdm import tqdm
import logging 
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def splice_rgb_patch_selection(config, patches, percentile_threshold):
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

def splice_features_patch_selection(config, patches, percentile_threshold):
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

def yottixel_rgb_patch_selection(config, patches, percentage_selected):	
    """
    Yottixel RGB Patch Selection
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
    kmeans_clusters = 9  # TODO: move to config if needed

    if len(patches) == 0:
        logging.warning("Empty patch list provided to Yottixel selection.")
        return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

    # ---- Stage 1: Color clustering ----
    rgb_hist = np.array([p['rgb_histogram'] for p in patches])
    kmeans_clusters = min(kmeans_clusters, len(patches))  # Cap clusters to number of patches
    kmeans_color = KMeans(
        n_clusters=kmeans_clusters,
        random_state=config["experiment"].get("random_state", None)
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
            random_state=config["experiment"].get("random_state", None)
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

def yottixel_features_patch_selection(config, patches, percentage_selected):
    """
    Yottixel-Features Patch Selection (RetCCL-Inspired)
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
    if len(patches) == 0:
        logging.warning("Empty patch list provided.")
        return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

    kmeans_clusters = 9  # TODO: move to config if needed

    # ---- Stage 1: Feature clustering ----
    features = np.array([p['feature'] for p in patches])
    kmeans_clusters = min(kmeans_clusters, len(patches))
    kmeans_feat = KMeans(
        n_clusters=kmeans_clusters,
        random_state=config["experiment"].get("random_state", None)
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
            random_state=config["experiment"].get("random_state", None)
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

def sdm_features_patch_selection(config, patches, percentage_selected=None):
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
    rng = np.random.default_rng(config.get("experiment", {}).get("random_state", None))
    selected = [int(rng.choice(idx_arr)) for idx_arr in groups.values()]

    # ---- Coords alongside groups ----
    coords = np.array([[int(p['loc'][0]), int(p['loc'][1])] for p in patches], dtype=int)

    return selected, group_ids, coords, groups