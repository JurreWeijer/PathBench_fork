
import logging 
import numpy as np
from sklearn.cluster import KMeans
import json
from types import SimpleNamespace
import pandas as pd
import os
import torch

from .base import MosaicSelector
from .registry import register_mosaic_selectors
from ..prototyping.model_factory import create_prototyping_model
from ..prototyping.tokenizer import PrototypeTokenizer

logger = logging.getLogger(__name__)

@register_mosaic_selectors
class PBMS(MosaicSelector):
    name = "pbms"
    HYPERPARAMS = {
        # (only the true hyperparams here; NOT prototypes_base / feature_string / in_dim)
        "model_type":   {"type": str,  "default": "PANTHER",         "attr": "model_type", "include_in_id":False},
        "model_config": {"type": str,  "default": "PANTHER_default", "attr": "model_config", "include_in_id": False},
        "n_proto":      {"type": int,  "default": 25, "min": 1,      "attr": "n_proto", "include_in_id": True, "id_order": 2},
        "n_per_proto":  {"type": int,  "default": 10000, "min": 1000,"attr": "n_per_proto", "include_in_id": True, "id_order": 3},
        "mode":         {"type": str,  "default": "kmeans",          "attr": "mode", "include_in_id": True, "id_order": 4},
        "normalization":{"type": bool, "default": False,             "attr": "normalization", "include_in_id": True, "id_order": 5},
        "out_type":     {"type": str,  "default": "allcat",          "attr": "out_type", "include_in_id": False},
        "em_iter":      {"type": int,  "default": 10, "min": 1,      "attr": "em_iter", "include_in_id": True, "id_order": 7},
        "tau":          {"type": float,"default": 10.0, "min": 0.0,  "attr": "tau", "include_in_id": False},
        "ot_eps":       {"type": float,"default": 0.1,  "min": 0.0,  "attr": "ot_eps", "include_in_id": False},
        "load_proto":   {"type": bool, "default": True,              "attr": "load_proto", "include_in_id": False},
        "fix_proto":    {"type": bool, "default": True,              "attr": "fix_proto", "include_in_id": False},
        "perc_selected":{"type": float,"default": 1.0, "min": 0.0, "max": 100.0, "attr": "perc_selected", "include_in_id": True, "id_order": 12},
        "save_slide_embeddings": {"type": bool, "default": False, "help": "If True, save PANTHER slide/prototype embeddings per slide.", "attr": "save_slide_embeddings", "include_in_id": False, "id_order": 99,},
    }

    def __init__(self, params, config, **kwargs):
        super().__init__(params, config, **kwargs)
        hp = self.hyperparam_spec()
        for k, meta in hp.items():
            setattr(self, meta.get("attr", k), self._get_hp(k))

        self.random_state  = (self.config.get("experiment", {}) or {}).get("random_state", None)

        # read extras from kwargs (NOT from params!)
        self.prototypes_base = kwargs.get("prototypes_base")
        self.feature_string  = kwargs.get("feature_string")
        self.in_dim          = kwargs.get("in_dim")

        self._last_slide_outputs = None

        if not self.prototypes_base or not self.feature_string or not self.in_dim:
            raise ValueError("PBMS requires 'prototypes_base', 'feature_string', and 'in_dim' via kwargs.")

        # reproduce producer folder name
        norm_suffix = "_norm" if self.normalization else ""
        prototype_id = f"{self.n_proto}_{self.feature_string}_{self.mode}_{self.n_per_proto}{norm_suffix}"
        self.prototype_folder = os.path.join(self.prototypes_base, "prototypes", prototype_id)
        self.embedding_folder = os.path.join(self.prototypes_base, "embeddings", prototype_id)

        self.prototype_path = os.path.join(self.prototype_folder, "prototypes.pkl")
        if not os.path.exists(self.prototype_path):
            raise FileNotFoundError(f"PBMS: prototypes not found at {self.prototype_path}")
        
        self.prototype_label_path = os.path.join(self.prototype_folder, "proto_labels.json")
        if not os.path.exists(self.prototype_label_path):
            raise FileNotFoundError(f"PBMS: prototype labels not found at {self.prototype_label_path}")

        self.model = create_prototyping_model(
            model_config=self.model_config,
            in_dim=self.in_dim,
            n_proto=self.n_proto,
            load_proto=self.load_proto,
            fix_proto=self.fix_proto,
            proto_path=self.prototype_path,
            out_type=self.out_type,
        )

        self._tokenizer = PrototypeTokenizer(self.model_type, self.out_type, self.n_proto)

        try:
            self.non_tumor_mask, self.proto_labels = self.load_non_tumor_mask()
        except Exception as e:
            raise ValueError(f"PBMS: failed to load non_tumor mask: {e}") from e

        if self.non_tumor_mask.shape[0] != self.n_proto:
            raise ValueError(
                f"PBMS: prototype mask size {self.non_tumor_mask.shape[0]} "
                f"!= n_proto {self.n_proto}"
            )
        
        
    def load_non_tumor_mask(self):
        with open(self.prototype_label_path, "r") as f:
            data = json.load(f)
        labels = data["label"]  # ensure your file has this
        mask = np.array([lab == "non_tumor" for lab in labels], dtype=bool)
        return mask, labels

    def _compute_resp(self, feats_np: np.ndarray) -> np.ndarray:
        """
        Given patch features [N,D], run model to get packed slide vector, tokenize to
        (prob, mean, var), then compute responsibilities resp [N,P].
        """
        device = "cuda" if (torch.cuda.is_available() and self.config.get("use_cuda", False)) else "cpu"
        x = torch.from_numpy(feats_np).float().unsqueeze(0).to(device)
        self.model.to(device).eval()
        with torch.no_grad():
            y = self.model(x)                 # [1, E]
        y = y.squeeze(0).detach().cpu()

        prob_t, mean_t, cov_t = self._tokenizer(y.unsqueeze(0))  # [1,P], [1,P,D], [1,P,D]
        prob = np.clip(prob_t.squeeze(0).cpu().numpy(), 1e-12, None).astype(np.float32)   # [P]
        mean = mean_t.squeeze(0).cpu().numpy().astype(np.float32)                         # [P,D]
        var  = np.clip(cov_t.squeeze(0).cpu().numpy().astype(np.float32), 1e-8, None)     # [P,D]

        z = feats_np.astype(np.float32)                      # [N,D]
        inv_var = 1.0 / var                                  # [P,D]
        log_pi  = np.log(prob)                                # [P]
        log_det = np.sum(np.log(2*np.pi*var), axis=1)         # [P]
        quad    = np.sum((z[:, None, :] - mean[None, :, :])**2 * inv_var[None, :, :], axis=2)  # [N,P]
        logN    = -0.5 * (quad + log_det[None, :])            # [N,P]
        log_post = log_pi[None, :] + logN                     # [N,P]
        m = np.max(log_post, axis=1, keepdims=True)
        resp = np.exp(log_post - (m + np.log(np.sum(np.exp(log_post - m), axis=1, keepdims=True) + 1e-12)))

        self._last_slide_outputs = {
            "slide_embed":  y.numpy().astype(np.float32),  # (E,)
            "proto_mean":   mean.astype(np.float32),       # (P,D)
            "proto_cov":    var.astype(np.float32),        # (P,D) diag
            "proto_prob":   prob.astype(np.float32),       # (P,)
        }

        return resp.astype(np.float32)  # [N,P]

    def export_slide_level(self, slide_id: str) -> None:
        """Save cached slide-level artifacts for this slide into save_dir."""
        data = self._last_slide_outputs or {}
        if not data:
            return
        np.savez_compressed(
            os.path.join(self.embedding_folder, f"{slide_id}.npz"),
            slide_embed=data.get("slide_embed", np.empty((0,), np.float32)),
            proto_mean=data.get("proto_mean",  np.empty((0, 0), np.float32)),
            proto_cov=data.get("proto_cov",    np.empty((0, 0), np.float32)),
            proto_prob=data.get("proto_prob",  np.empty((0,), np.float32)),
        )

    def run(self, patches, **_):
        """
        Input:
            patches: list of dicts for ONE slide, each dict has:
              - 'feature': np.ndarray(D,)
              - 'loc':     (x, y) coords (level-0)
        Returns:
            selected_indices: list[int]
            group_ids:        np.ndarray[int] length N; 0 for excluded prototypes,
                              else (proto_id + 1) for included prototypes.
            coords:           np.ndarray[int] shape [N,2]
            groups:           dict[int, np.ndarray] mapping group_id -> indices
        """
        if len(patches) == 0:
            logging.warning("Empty patch list provided to PBPS.")
            return [], np.array([], dtype=int), np.empty((0, 2), dtype=int), {}

        # Assemble features and coords
        feats = np.asarray([p["feature"] for p in patches], dtype=float)  # [N,D]
        coords = np.array([[int(p["loc"][0]), int(p["loc"][1])] for p in patches], dtype=int)  # [N,2]

        if feats.ndim != 2 or feats.shape[1] != self.in_dim:
            raise RuntimeError(f"Feature shape mismatch: got {feats.shape}, expected [N,{self.in_dim}]")

        # Responsibilities and hard assignment to prototypes
        resp = self._compute_resp(feats)             # [N,P]
        top1 = np.argmax(resp, axis=1)
        
        # Map to group ids: excluded proto -> 0, included -> proto_id+1
        included_mask = ~self.non_tumor_mask         # True means tumor-capable proto
        group_ids = np.where(included_mask[top1], top1 + 1, 0).astype(int)  # [N]

        # Build groups dict (keys are group ids actually present)
        groups = {}
        for gid in np.unique(group_ids):
            groups[int(gid)] = np.where(group_ids == gid)[0]

        # Selection: only within included prototypes (gid >= 1), per-group spatial KMeans,
        # selecting a percentage of representatives per prototype like Yottixel.
        selected = []
        perc = float(self.perc_selected)

        for gid in sorted(k for k in groups.keys() if k >= 1):  # skip 0 (excluded protos)
            member_idx = groups[gid]
            if member_idx.size == 0:
                continue

            n_select = max(1, int(len(member_idx) * (perc / 100.0)))

            # Special-case: if n_select == 1, pick the first (consistent with Yottixel behavior)
            if n_select == 1:
                selected.append(int(member_idx[0]))
                continue

            loc_features = coords[member_idx].astype(float)  # [M,2]
            kmeans_loc = KMeans(n_clusters=n_select, random_state=self.random_state)
            dists = kmeans_loc.fit_transform(loc_features)   # [M, n_select]

            used_local = set()
            for c in range(n_select):
                # pick nearest unused to center c
                sorted_local = np.argsort(dists[:, c])
                for sidx in sorted_local:
                    if sidx not in used_local:
                        used_local.add(sidx)
                        selected.append(int(member_idx[sidx]))
                        break

        return selected, group_ids, coords, groups

    def additional_data(self) -> dict:
        # Use the last computed slide outputs (set during run)
        d = getattr(self, "_last_slide_outputs", None) or {}
        # Make sure it’s pickle-friendly (convert torch → numpy if needed)
        out = {}
        for k in ("slide_embed", "proto_mean", "proto_cov", "proto_prob"):
            v = d.get(k)
            if v is None:
                continue
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            out[k] = np.asarray(v)
        # You can also drop small bits of metadata for traceability
        out["pbms_meta"] = {
            "n_proto": int(self.n_proto),
            "mode": str(self.mode),
            "normalization": bool(self.normalization),
            "feature_string": str(self.feature_string) if hasattr(self, "feature_string") else None,
            "proto_labels": list(self.proto_labels)
        }
        return {"prototyping": out}
        
