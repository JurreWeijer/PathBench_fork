# ----------------------------------------------------------------------
# RetCCLDatabase
#   Same public API as YottixelDatabase:
#       __init__(config, slide_representation_paths, k=3, cosine_threshold=0.7)
#       predict_slide(query_bob)  -> dict
#       leave_one_patient_out()   -> list[dict]
#
#   Internal helper class RetCCLSlide keeps slide data together
#   (slide_id, patient_id, label, features), so no scattered lists.
# ----------------------------------------------------------------------

import os
from os.path import splitext
from statistics import mode, mean
from collections import Counter
import numpy as np
from numpy.linalg import norm
import pandas as pd
import torch
from tqdm import tqdm

from ...utils import load_patch_dicts_pickle  # same as your Yottixel file
from ..registry import register_search_methods
from ..base import SearchMethodBase

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def _safe_mean(vals):
    return float(np.mean(vals)) if len(vals) else 0.0


class RetCCLSlide:
    """Simple holder for one slide's data (BoB-like, but for RETCCL)."""
    def __init__(self, slide_id, patient_id, label, features: np.ndarray):
        self.slide_id = slide_id
        self.patient_id = patient_id
        self.label = label
        self.features = features  # (n_patches, d) or (1, d)

@register_search_methods
class RetCCLSearch(SearchMethodBase):

    """
    RETCCL-based retrieval with YottixelDatabase-compatible I/O.
    No separate organ/tumor branches; single path as requested.
    """
    name = "retccl"
    supports = {"patch"}

    HYPERPARAMS = {
        "k": {"type": int, "default": 5, "min": 1, 
              "help": "retrieval depth", 
              "attr": "k",
              "include_in_id": True, "id_order": 0
              },
        "cosine_threshold": {"type": float, "default": 0.7, "min": 0.0, "max": 1.0,
                             "help": "min cosine similarity to accept a patch match",
                             "attr": "cosine_threshold",
                             "include_in_id": True, "id_order": 1
                            },
        "class_weight_factor":{"type": float, "default": 10.0, "min": 0.0,
                               "help": "inverse-frequency reweighting strength",
                               "attr": "class_weight_factor",
                               "include_in_id": True, "id_order": 2
                            },
        "topk_per_patch":{"type": int,   "default": 5, "min": 1,
                          "help": "how many top sims per patch to use",
                          "attr": "topk_per_patch",
                          "include_in_id": True, "id_order": 3
                        },
    }
    
    def __init__(self, config: dict, slide_representation_paths: dict, params: dict, **kwargs):
        # Base resolves ALL HYPERPARAMS and attaches them as attributes (incl. self.k)
        super().__init__(config=config, slide_representation_paths=slide_representation_paths, params=params, **kwargs)

        self.is_slide = (self.mode == "slide")  # always False with supports={"patch"}, kept for consistency

        # Annotations
        ann_path = self.config['experiment']['annotation_file']
        self.annotations = pd.read_csv(ann_path).set_index("slide")

        # Build in-memory index (patch features per slide)
        self.slide_index = {}  # slide_id -> RetCCLSlide
        self._build_patch_features(self.paths)

        # Class weights like your original
        self.class_weight = self._compute_class_weights(factor=self.class_weight_factor)

    # ------------------------------
    # Builders
    # ------------------------------
    def _build_patch_features(self, slide_representation_paths: dict):
        for slide_id, mosaic_pkl in tqdm(slide_representation_paths.items(), desc="Building RETCCL patch feats"):
            if slide_id not in self.annotations.index:
                continue

            label = self.annotations.loc[slide_id]["category"]
            patient_id = self.annotations.loc[slide_id]["patient"]

            mosaic_data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=True)
            patches = mosaic_data["patches"]
            feats = np.stack([p['feature'] for p in patches], axis=0)  # (n_patches, d)

            s = RetCCLSlide(slide_id, patient_id, label, feats)
            self.slide_index[slide_id] = s
    
    def _compute_class_weights(self, factor: int = 10):
        """
        Mirror RETCCL: weight[label] ∝ 1 / count(label), then normalized so the sum ≈ factor.
        """
        label_counts = Counter(s.label for s in self.slide_index.values())
        inv_sum = sum(1.0 / c for c in label_counts.values())
        norm_factor = factor / inv_sum
        return {lbl: norm_factor * (1.0 / cnt) for lbl, cnt in label_counts.items()}

    # ------------------------------
    # Public API
    # ------------------------------
    def predict_slide(self, q_slide: RetCCLSlide) -> dict:
        """
        Accepts the same BoB object you pass to YottixelDatabase.predict_slide.
        Uses its slide_id/label/patient_id to fetch our RetCCLSlide.
        """
        q_id = q_slide.slide_id
        q_label = q_slide.label
        q_patient = q_slide.patient_id
        q_feats = q_slide.features  # (n_q, d)

        flat_feats = []
        flat_meta  = []  # (slide_id, patch_idx)

        for s in self.slide_index.values():
            if s.slide_id == q_id or s.patient_id == q_patient:
                continue
            f = s.features
            for pidx in range(f.shape[0]):
                flat_feats.append(f[pidx])
                flat_meta.append((s.slide_id, pidx))

        if not flat_feats:  # nothing to compare against
            return {
                "query_slide_id": q_id,
                "query_label": q_label,
                "predicted_label": q_label,
                "top_k": []
            }

        flat_feats = np.asarray(flat_feats)
        flat_norms = norm(flat_feats, axis=1) + 1e-12  # precompute once

        Bag = {}
        Entropy = {}
        for patch_idx, qf in enumerate(q_feats):
            qn = norm(qf) + 1e-12
            sims = (flat_feats @ qf) / (flat_norms * qn)

            mask = sims >= self.cosine_threshold
            idxs = np.where(mask)[0]
            bag = [(int(i), float(sims[i])) for i in idxs]
            bag.sort(key=lambda x: x[1], reverse=True)
            Bag[patch_idx] = bag

            if not bag:
                Entropy[patch_idx] = 0.0
                continue

            # Weighted entropy
            weight_sums = {}
            total_w = 0.0
            for flat_idx, sim in bag:
                slide_id, _ = flat_meta[flat_idx]
                label = self.slide_index[slide_id].label
                w_sim = ((sim + 1.0) / 2.0) * self.class_weight[label]
                weight_sums[label] = weight_sums.get(label, 0.0) + w_sim
                total_w += w_sim

            entropy = 0.0
            for w in weight_sums.values():
                p = w / total_w
                entropy -= p * np.log(p)
            Entropy[patch_idx] = entropy

        # Order by entropy desc
        Bag = dict(sorted(Bag.items(), key=lambda x: Entropy[x[0]], reverse=True))

        # eta threshold
        kq = q_feats.shape[0]                  # number of query patches
        eta_threshold = 0.0
        for idx in range(kq):
            top5 = [s for (_, s) in Bag.get(idx, [])[:self.topk_per_patch]]   # top-5 sims for this patch
            eta_threshold += _safe_mean(top5)                # add this patch's mean(top5)
        eta_threshold /= max(kq, 1)                          # average over all patches

        # filter bags
        drop_ids = []
        for idx, bag in Bag.items():
            if _safe_mean([s for (_, s) in bag[:self.topk_per_patch]]) < eta_threshold:
                drop_ids.append(idx)
        for idx in drop_ids:
            del Bag[idx]

        # majority vote per bag -> slide list
        WSIRet = {}
        for _, bag in Bag.items():
            top5 = bag[:self.topk_per_patch]
            if not top5:
                continue

            match_labels, match_slides, sims = [], [], []
            for i, sim in top5:
                sid, _ = flat_meta[i]
                match_labels.append(self.slide_index[sid].label)
                match_slides.append(sid)
                sims.append(sim)

            mlabel = mode(match_labels)
            chosen_slide = match_slides[match_labels.index(mlabel)]
            if chosen_slide not in WSIRet:
                WSIRet[chosen_slide] = (chosen_slide, sims[match_labels.index(mlabel)], mean(sims))

        # Convert to Yottixel top_k format
        wsiret_sorted = sorted(WSIRet.values(), key=lambda x: x[2], reverse=True)
        top_k_info = [{
            "slide_id": sid,
            "label": self.slide_index[sid].label,
            "distance": float(1.0 - avg_sim)  # align "distance" to smaller=closer
        } for sid, _, avg_sim in wsiret_sorted[: self.k]]

        predicted = Counter([t["label"] for t in top_k_info]).most_common(1)[0][0] if top_k_info else q_label

        return {
            "query_slide_id": q_id,
            "query_label": q_label,
            "predicted_label": predicted,
            "top_k": top_k_info
        }

    def leave_one_patient_out(self):
        """
        Identical output shape to YottixelDatabase.leave_one_patient_out()
        We can just pass our RetCCLSlide objects directly (they have the attrs we need).
        """
        results = []
        for s in tqdm(self.slide_index.values(), desc="RETCCL LOPO retrieval"):
            # s has slide_id, patient_id, label; predict_slide accepts anything with those attrs
            results.append(self.predict_slide(s))
        return results
