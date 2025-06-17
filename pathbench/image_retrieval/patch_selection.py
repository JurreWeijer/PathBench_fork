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
from slideflow.model import build_feature_extractor

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

    if percentile_threshold is None:
        raise ValueError("percentile_threshold must be specified for SPLICE.")

    # Extract color features (normalized RGB histograms)
    color_features = np.array([patch['rgb_histogram'] for patch in patches])

    num_patches = color_features.shape[0]
    selected_indices = []
    excluded = np.zeros(num_patches, dtype=bool)  # Tracks which patches are excluded

    for i in range(num_patches):
        if excluded[i]:
            continue

        ref_feat = color_features[i]
        remaining_idx = np.where(~excluded)[0]  # Indices of remaining (not yet excluded) patches

        # Compute distance from current patch to all others
        distances = np.linalg.norm(color_features[remaining_idx] - ref_feat, axis=1)

        # Determine suppression threshold based on percentile
        thresh = np.percentile(distances, percentile_threshold)

        # Exclude nearby redundant patches
        for j, d in zip(remaining_idx, distances):
            if j == i:
                continue
            if d < thresh:
                excluded[j] = True

        selected_indices.append(i)

    return selected_indices

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
        patches (list of dict): List of patch dictionaries, each containing:
            - 'features': Deep feature embedding of the patch.
        percentile_threshold (float): Percentile value (0-100) for distance suppression threshold.

    Returns:
        list: Indices of selected patches from the input list.

    Reference:
        Alsaafin, Areej, Peyman Nejat, Abubakr Shafique, Jibran Khan, Saghir Alfasly, Ghazal Alabtah, and H. R. Tizhoosh. 
        “SPLICE -- Streamlining Digital Pathology Image Processing.” arXiv, April 26, 2024. https://doi.org/10.48550/arXiv.2404.17704.
    """

    if percentile_threshold is None:
        raise ValueError("percentile_threshold must be specified for SPLICE.")

    # Extract feature vectors from all patches
    features = np.array([patch['feature'] for patch in patches])

    num_patches = features.shape[0]
    selected_indices = []
    excluded = np.zeros(num_patches, dtype=bool)

    for i in range(num_patches):
        if excluded[i]:
            continue

        ref_feat = features[i]
        remaining_idx = np.where(~excluded)[0]

        # Compute distance from reference patch to remaining patches
        distances = np.linalg.norm(features[remaining_idx] - ref_feat, axis=1)

        # Compute suppression threshold based on user-defined percentile
        thresh = np.percentile(distances, percentile_threshold)

        # Exclude similar patches
        for j, d in zip(remaining_idx, distances):
            if j == i:
                continue
            if d < thresh:
                excluded[j] = True

        selected_indices.append(i)

    return selected_indices

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
        patches (list of dict): List of patch dictionaries, each containing:
            - 'rgb_histogram': Normalized RGB histogram.
            - 'wsi_loc': (x, y) location of the patch in the WSI.
        percentage_selected (float): Percentage of patches to select from each color cluster.

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
        return []

    # ---- Stage 1: Color clustering ----
    rgb_hist = np.array([p['rgb_histogram'] for p in patches])
    kmeans_clusters = min(kmeans_clusters, len(patches))  # Cap clusters to number of patches
    kmeans_color = KMeans(n_clusters=kmeans_clusters, random_state=config["experiment"].get("random_state", None))
    color_labels = kmeans_color.fit_predict(rgb_hist)

    selected_indices = []
    for i in range(kmeans_clusters):
        # Get all patches belonging to color cluster i
        cluster_patches = [p for p, lbl in zip(patches, color_labels) if lbl == i]
        if len(cluster_patches) == 0:
            continue

        n_select = max(1, int(len(cluster_patches) * percentage_selected / 100))

        # ---- Stage 2: Spatial clustering ----
        loc_features = [p['wsi_loc'] for p in cluster_patches]
        kmeans_loc = KMeans(n_clusters=n_select, random_state=config["experiment"].get("random_state", None))
        dists = kmeans_loc.fit_transform(loc_features)

        used = set()
        for idx in range(n_select):
            # For each cluster center, find closest unused patch
            sorted_idx = np.argsort(dists[:, idx])
            for sidx in sorted_idx:
                if sidx not in used:
                    used.add(sidx)
                    selected_indices.append(patches.index(cluster_patches[sidx]))
                    break

    return selected_indices

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
        patches (list of dict): List of patch dictionaries, each containing:
            - 'features': Deep feature embedding of the patch.
            - 'wsi_loc': (x, y) location of the patch in the WSI.
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
        return []
    
    kmeans_clusters = 15  # TODO: move to config if needed

    # ---- Stage 1: Feature clustering ----
    features = np.array([p['feature'] for p in patches])
    kmeans_clusters = min(kmeans_clusters, len(patches))
    kmeans_feat = KMeans(n_clusters=kmeans_clusters, random_state=config["experiment"].get("random_state", None))
    feat_labels = kmeans_feat.fit_predict(features)

    selected_indices = []
    for i in range(kmeans_clusters):
        cluster_patches = [p for p, label in zip(patches, feat_labels) if label == i]
        if len(cluster_patches) == 0:
            continue

        n_select = max(1, int(len(cluster_patches) * percentage_selected / 100))

        # If only one patch should be selected, skip spatial clustering
        if n_select == 1:
            selected_indices.append(patches.index(cluster_patches[0]))
            continue

        # ---- Stage 2: Spatial clustering ----
        locs = np.array([p['wsi_loc'] for p in cluster_patches])
        kmeans_loc = KMeans(n_clusters=n_select, random_state=config["experiment"].get("random_state", None))
        dists = kmeans_loc.fit_transform(locs)

        used = set()
        for idx in range(n_select):
            # For each cluster center, select the closest unused patch
            sorted_idx = np.argsort(dists[:, idx])
            for sidx in sorted_idx:
                if sidx not in used:
                    used.add(sidx)
                    selected_indices.append(patches.index(cluster_patches[sidx]))
                    break

    return selected_indices

def sdm_features_patch_selection(config, patches, percentile):
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
        list of int:
            Indices of `patches` selected by SDM (one per distance bin).

    Reference:
        Shafique, Salman, et al.  
        “Selection of Distinct Morphologies to Divide & Conquer Gigapixel Pathology Images.”  
        Medical Image Analysis (2023). https://doi.org/10.1016/j.media.2023.101757
    """

    if not patches:
        logging.warning("Empty patch list provided to SDM.")
        return []

    # ---- Stack features and compute centroid ----
    feats = np.stack([p['feature'] for p in patches], axis=0)
    centroid = feats.mean(axis=0)

    # ---- Compute distances and discretize into integer bins ----
    dists = np.linalg.norm(feats - centroid[None, :], axis=1)
    bins = np.rint(dists).astype(int)

    # ---- Randomly pick one patch per bin ----
    rng = np.random.default_rng(config["experiment"].get("random_state", None))
    selected = []
    for bin_id in np.unique(bins):
        candidates = np.where(bins == bin_id)[0]
        chosen = int(rng.choice(candidates))
        selected.append(chosen)

    return selected

"""def splice_patch_selection(patches, features, percentile_threshold):
    
    logging.info("Starting SPLICE patch selection...")

    if percentile_threshold is None: 
        raise ValueError("percentile_threshold must be specified for SPLICE.")
    
    color_features = np.array([np.mean(p.reshape(-1, p.shape[-1]), axis=0) for p in patches])

    num_patches = color_features.shape[0]
    selected_indices = []
    excluded = np.zeros(num_patches, dtype=bool)

    for i in tqdm(range(num_patches), desc="SPLICE patch selection", leave=False):
        if excluded[i]:
            continue
        ref_feat = color_features[i]
        remaining_idx = np.where(~excluded)[0]
        distances = np.linalg.norm(color_features[remaining_idx] - ref_feat, axis=1)
        thresh = np.percentile(distances, percentile_threshold)
        for j, d in zip(remaining_idx, distances):
            if j == i:
                continue
            if d < thresh:
                excluded[j] = True
        selected_indices.append(i)
    logging.info("SPLICE patch selection completed.")

    return selected_indices"""