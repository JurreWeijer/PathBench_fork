import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from ...utils import load_patch_dicts_pickle
from ..base import SearchMethodBase
from ..registry import register_search_methods

logger = logging.getLogger(__name__)

def _per_proto_diag_mahalanobis(mu_a, var_a, mu_b, var_b, eps=1e-6):
    """Pairwise same-prototype distance with diag cov: sqrt( sum_j (Δ^2 / (σ_a^2+σ_b^2+eps)) )."""
    # Ensure same shape (P, D)
    P = min(mu_a.shape[0], mu_b.shape[0])
    D = min(mu_a.shape[1], mu_b.shape[1])
    mu_a = mu_a[:P, :D]; mu_b = mu_b[:P, :D]
    var_a = var_a[:P, :D]; var_b = var_b[:P, :D]

    s = var_a + var_b + eps
    diff = mu_a - mu_b
    d2 = np.sum((diff * diff) / s, axis=1)
    return np.sqrt(np.maximum(d2, 0.0))  # (P,)

def _softmin(distances, tau=0.5):
    """Soft-min aggregation: -tau * log(mean(exp(-(d - min)/tau))) + min."""
    d = np.asarray(distances, dtype=np.float32)
    m = np.min(d)
    z = np.exp(-(d - m) / max(tau, 1e-8))
    return float(-tau * np.log(np.mean(z)) + m)

@register_search_methods
class PrototypeSimilaritySearch(SearchMethodBase):
    """
    Variance-aware prototype distance (same-ID only), using optional per-slide
    prototype labels read from the mosaic .pkl:

        mosaic["prototyping"]["proto_labels"] == list[str]   # e.g., "non_tumor", "tumor", ...

    If `use_proto_labels=True`, prototypes labeled "non_tumor" are excluded.
    The comparison uses the AND of masks from query and candidate slides.
    """
    name = "pbss"
    supports = {"patch"}

    HYPERPARAMS = {
        "k":   {"type": int,   "default": 10, "min": 1,
                "help": "retrieval depth (top-k)", "attr": "k",
                "include_in_id": True, "id_order": 0},
        "eps": {"type": float, "default": 1e-6, "min": 1e-12,
                "help": "variance floor in Mahalanobis", "attr": "eps",
                "include_in_id": True, "id_order": 1},
        "tau": {"type": float, "default": 0.5, "min": 1e-4,
                "help": "soft-min temperature", "attr": "tau",
                "include_in_id": True, "id_order": 2},
        "use_proto_labels": {"type": bool, "default": True,
                "help": "If True, use mosaic['prototyping']['proto_labels'] to exclude 'non_tumor' prototypes.",
                "attr": "use_proto_labels",
                "include_in_id": True, "id_order": 3},
    }

    def __init__(self, config: dict, slide_representation_paths: dict, params: dict, **kwargs):
        super().__init__(config=config,
                         slide_representation_paths=slide_representation_paths,
                         params=params,
                         **kwargs)
        self.k   = self._get_hp("k")
        self.eps = self._get_hp("eps")
        self.tau = self._get_hp("tau")
        self.use_proto_labels = self._get_hp("use_proto_labels")

        # Load annotations
        ann_path = self.config['experiment']['annotation_file']
        self.annotations = pd.read_csv(ann_path).set_index("slide")

        # Load slide entries: mu, var, and optional per-slide include_mask
        self._entries = []
        for slide_id, mosaic_pkl in tqdm(slide_representation_paths.items(), desc="Loading prototype reps"):
            if slide_id not in self.annotations.index:
                continue

            label      = self.annotations.loc[slide_id]["category"]
            patient_id = self.annotations.loc[slide_id].get("patient", None)

            try:
                data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=False)
            except Exception as e:
                logger.warning(f"Failed to load {mosaic_pkl}: {e}")
                continue

            proto = (data or {}).get("prototyping", None)
            if proto is None:
                logger.warning(f"No 'prototyping' block in {mosaic_pkl}; skipping {slide_id}.")
                continue

            mu  = np.asarray(proto.get("proto_mean"), dtype=np.float32)
            var = np.asarray(proto.get("proto_cov"),  dtype=np.float32)
            if mu.ndim != 2 or var.ndim != 2:
                logger.warning(f"Bad proto shapes for {slide_id}; skipping.")
                continue

            # Optional per-slide labels → mask
            include_mask = None
            if self.use_proto_labels:
                labels = proto.get("proto_labels", None)
                if isinstance(labels, list):
                    # assume labels align with prototypes index
                    labels_lower = [str(x).lower() for x in labels]
                    if len(labels_lower) >= mu.shape[0]:
                        include_mask = np.array([lab != "non_tumor" for lab in labels_lower[:mu.shape[0]]],
                                                dtype=bool)
                        # If mask removes everything, ignore it.
                        if not np.any(include_mask):
                            logger.warning(f"All prototypes excluded by labels for {slide_id}; ignoring label mask.")
                            include_mask = None
                    else:
                        logger.warning(
                            f"proto_labels length {len(labels_lower)} < P={mu.shape[0]} for {slide_id}; ignoring labels."
                        )

            self._entries.append({
                "slide_id": slide_id,
                "patient":  patient_id,
                "label":    label,
                "mu":       mu,
                "var":      var,
                "mask":     include_mask  # may be None
            })

        if not self._entries:
            raise RuntimeError("No valid slides with prototyping data were loaded.")

        # Determine a common prototype count (min over slides), for alignment safety.
        self._P_common = min(e["mu"].shape[0] for e in self._entries)

    def _apply_masks_and_clip(self, mu, var, mask):
        """Clip to common P, then apply mask if provided."""
        mu = mu[:self._P_common, :]
        var = var[:self._P_common, :]
        if mask is not None:
            mu = mu[mask[:self._P_common]]
            var = var[mask[:self._P_common]]
        return mu, var

    def _distance(self, a, b) -> float:
        # Compose joint mask: AND of both masks if both present; else whichever exists; else None.
        joint_mask = None
        if a["mask"] is not None and b["mask"] is not None:
            # Align both to _P_common
            joint_mask = a["mask"][:self._P_common] & b["mask"][:self._P_common]
            if not np.any(joint_mask):
                joint_mask = None  # fallback to no mask
        elif a["mask"] is not None:
            joint_mask = a["mask"][:self._P_common]
            if not np.any(joint_mask):
                joint_mask = None
        elif b["mask"] is not None:
            joint_mask = b["mask"][:self._P_common]
            if not np.any(joint_mask):
                joint_mask = None

        mu_a, var_a = self._apply_masks_and_clip(a["mu"], a["var"], joint_mask)
        mu_b, var_b = self._apply_masks_and_clip(b["mu"], b["var"], joint_mask)

        # If (paranoia) masking removed all prototypes, fall back to unmasked.
        if mu_a.size == 0 or mu_b.size == 0:
            mu_a, var_a = a["mu"][:self._P_common], a["var"][:self._P_common]
            mu_b, var_b = b["mu"][:self._P_common], b["var"][:self._P_common]

        d_p = _per_proto_diag_mahalanobis(mu_a, var_a, mu_b, var_b, eps=self.eps)  # (P_eff,)
        return _softmin(d_p, tau=self.tau)

    def _predict_one(self, q_entry):
        # LOPO: exclude same patient when available
        atlas = [e for e in self._entries if e["slide_id"] != q_entry["slide_id"]
                 and (e["patient"] is None or q_entry["patient"] is None or e["patient"] != q_entry["patient"])]

        if not atlas:
            return {
                "query_slide_id": q_entry["slide_id"],
                "query_label": q_entry["label"],
                "predicted_label": q_entry["label"],
                "top_k": []
            }

        distances = np.array([self._distance(q_entry, e) for e in atlas], dtype=np.float32)
        order = np.argsort(distances)[: self.k]
        top_k = [{"slide_id": atlas[i]["slide_id"],
                  "label":    atlas[i]["label"],
                  "distance": float(distances[i])}
                 for i in order]

        # majority vote
        labels = [x["label"] for x in top_k]
        uniq, counts = np.unique(labels, return_counts=True)
        maj = uniq[np.argmax(counts)]

        return {
            "query_slide_id": q_entry["slide_id"],
            "query_label": q_entry["label"],
            "predicted_label": str(maj),
            "top_k": top_k
        }

    def leave_one_patient_out(self):
        results = []
        for q in tqdm(self._entries, desc="LOPO (proto_mvn)"):
            results.append(self._predict_one(q))
        return results
