import os
import json
import gc
import logging
import pickle
from collections import OrderedDict, defaultdict, Counter
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import psutil
from tqdm import tqdm
from torchvision import transforms

import torch.multiprocessing as tmp
tmp.set_sharing_strategy("file_system")

from ..sish.sish_index import compute_latent_features, slide_to_index
from ..sish.sish_eval import Uncertainty_Cal, Clean, Filtered_BY_Prediction
from ..sish.sish_vqvae import LargeVectorQuantizedVAE_Encode
from ..sish.sish_veb import VEB
from ...utils import load_patch_dicts_pickle

from ..registry import register_search_methods
from ..base import SearchMethodBase

logger = logging.getLogger(__name__)

# must live at module scope so pickle can find it
def scale_to_minus1_to_1(x):
    """
    Transform tensor values in [0,1] into [-1,1] for the VQ‑VAE.
    """
    return 2 * x - 1

def hamming_bytes(a: bytes, b: bytes) -> int:
    """Fast Hamming distance between two equal‑length byte strings."""
    arr_a = np.frombuffer(a, dtype=np.uint64, count=len(a)//8)
    arr_b = np.frombuffer(b, dtype=np.uint64, count=len(b)//8)
    return int(sum(int(v).bit_count() for v in (arr_a ^ arr_b)))

def log_mem(tag):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss/1e9
    kids = sum((c.memory_info().rss for c in p.children(recursive=True)
                if c.is_running()), 0)/1e9
    shm  = psutil.disk_usage('/dev/shm').used/1e9 if os.path.exists('/dev/shm') else 0
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated()/1e6
        reserv= torch.cuda.memory_reserved()/1e6
    else:
        alloc = reserv = 0
    logging.info(f"[{tag}] RSS={rss:.2f} GB  kids={kids:.2f} GB  "
                 f"CUDA alloc/res={alloc:.0f}/{reserv:.0f} MB  /dev/shm={shm:.2f} GB")

@register_search_methods
class SISHSearch(SearchMethodBase):
    """
    SISHDatabase implements the Selection of Informative Samples in Histopathology (SISH) retrieval pipeline.
    It builds and manages an index over slide-level patch mosaics using a VQ-VAE encoder,
    hierarchical pooling, and a Van Emde Boas (VEB) tree for efficient nearest-neighbor search.
    """
    name = "sish"
    supports = {"patch"}

    HYPERPARAMS = {
        # retrieval depth (kept for consistency across methods)
        "k":                {"type": int, "default": 10, "min": 1,   
                             "help": "top-k retrieval depth", 
                             "attr": "k",
                             "include_in_id": True, "id_order": 0,
                             },
        # SISH traversal knobs
        "seed_interval_c":  {"type": int, "default": 50, "min": 1,   
                             "help": "seed index stride",
                             "attr": "seed_interval_c",
                             "include_in_id": True, "id_order": 1,
                             },
        "seed_fanout_t":    {"type": int, "default": 10, "min": 1,   
                             "help": "seeds per side",
                             "attr": "seed_fanout_t",
                             "include_in_id": True, "id_order": 2,
                             },
        "pre_step":         {"type": int, "default": 375, "min": 0,   
                             "help": "max predecessor steps",
                             "attr": "pre_step",
                             "include_in_id": True, "id_order": 3,
                             },
        "succ_step":        {"type": int, "default": 375, "min": 0,
                             "help": "max successor steps",
                             "attr": "succ_step",
                             "include_in_id": True, "id_order": 4,
                             },
        "hamming_thr":      {"type": int, "default": 512, "min": 0,   
                             "help": "acceptance threshold",
                             "attr": "hamming_thr",
                             "include_in_id": True, "id_order": 5,
                             },
        # index build / output behavior
        "resume_shards":    {"type": bool, "default": True,
                             "help": "resume shard building",
                             "attr": "resume_shards",
                             "include_in_id": False, "id_order": 6,
                            },
        "shard_size":       {"type": int, "default": 25, "min": 1,
                             "help": "slides per shard",
                             "attr": "shard_size",
                             "include_in_id": False, "id_order": 7,
                            },
        "return_patch_matches": {"type": bool, "default": False,
                                 "help": "emit patch-level matches",
                                 "attr": "return_patch_matches",
                                 "include_in_id": False, "id_order": 8,
                                 },
        }

    def __init__(self, config: dict, slide_representation_paths: Dict[str, str], params: Dict, **kwargs) -> None:
        """
        Initialize the SISHDatabase.

        Args:
            config (dict): Experiment configuration dictionary.
            slide_mosaic_paths (Dict[str, str]): Mapping from slide ID to mosaic .pkl file paths.
            k (int): Number of nearest neighbors (top-k) to retrieve.
            mosaic_string (str): Identifier for the mosaic variant (e.g., 'SPLICE_rgb-25_uni').

        Attributes:
            meta (Dict[int, List[dict]]): In-memory metadata: keys to patch entries.
            keys (List[int]): Flat list of integer indices for VEB insertion.
            vebtree (VEB): Van Emde Boas tree instance.
            is_patch (bool): Mode flag (slide vs. patch level).
            index_veb_path (str): Disk path to save/load the VEB tree.
            meta_database_path (str): Disk path to save/load metadata.
            annotations (DataFrame): Slide-level annotation table.
            vqvae (LargeVectorQuantizedVAE_Encode): Encode-only VQ-VAE model.
            transform_vqvqe (Callable): Transform mapping images to [-1,1] tensor.
            pool_layers (List[nn.Module]): Pooling layers for hierarchical sums.
            pool (Pool): Multiprocessing pool for semantic mapping.
        """
        super().__init__(config=config, slide_representation_paths=slide_representation_paths, params=params, **kwargs)
                # Keep your existing names so downstream code stays unchanged
        
        try:
            self.mosaic_string = self.params["mosaic_string"]
        except KeyError:
            raise ValueError("SISHSearch requires 'mosaic_string' in params (not a hyperparam).")

        if self.return_patch_matches:
            raise NotImplementedError(
                "return_patch_matches=True is not supported: 'patch_name' is not stored in meta."
            )

        # Derive the old is_patch flag from the validated mode
        self.is_slide = (self.mode == "slide") 

        # ---- create directories for index storage ----
        project_dir = os.path.join("experiments", self.config['experiment']['project_name'])
        sish_dir = os.path.join(project_dir, "sish")
        os.makedirs(sish_dir, exist_ok=True)
        self.index_veb_path = os.path.join(sish_dir, f"veb_{self.mosaic_string}.pkl")
        self.meta_database_path = os.path.join(sish_dir, f"meta_{self.mosaic_string}.pkl")

        # ---- load slide-level annotations ----
        annotations_path = self.config['experiment']['annotation_file']
        self.annotations = pd.read_csv(annotations_path).set_index('slide')
        logging.info(f"Loaded annotations for {len(self.annotations)} slides from {annotations_path}")

        # ---- SISH metrics paths; heavy objects deferred ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._codebook_path   = None
        self._checkpoint_path = None

        self.codebook_semantic = None
        self.vqvae = None
        self.transform_vqvqe = None

        # ---- setup pooling layers for hierarchical sums ----
        self.pool_layers = [
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.AvgPool2d((2, 2))
        ]

        # ---- in-memory structures ----
        self.meta = {}   # key:int -> List[meta-dict]
        self.keys = []   # flat list of all keys for VEB
        self.vebtree = None
        self.pool = None
    
    def _ensure_vqvae_ready(self) -> None:
        """Lazy-load the SISH VQ-VAE encoder, codebook, and transforms if not already loaded."""
        if self.vqvae is not None and self.codebook_semantic is not None and self.transform_vqvqe is not None:
            return
        
        sish_metrics = self.config["experiment"].get("SISH_metrics", {})
        
        self._codebook_path   = sish_metrics.get("codebook_semantic")
        self._checkpoint_path = sish_metrics.get("vqvae_checkpoint")
        if self._codebook_path is None or self._checkpoint_path is None:
            raise RuntimeError("SISH required but missing paths in experiment.SISH_metrics (need 'codebook_semantic' and 'vqvae_checkpoint').")

        # Load codebook (CPU is fine; it’s small)
        self.codebook_semantic = torch.load(self._codebook_path, map_location="cpu")
        logging.debug(f"Loaded semantic codebook from {self._codebook_path}")

        # Create encoder and load only encoder+codebook weights from checkpoint
        self.vqvae = LargeVectorQuantizedVAE_Encode(code_dim=256, code_size=128)
        raw = torch.load(self._checkpoint_path, map_location="cpu")["model"]
        enc_weights = OrderedDict({
            k[len("module."):]: v
            for k, v in raw.items()
            if k.startswith("module.encoder.") or k.startswith("module.codebook.")
        })
        missing, unexpected = self.vqvae.load_state_dict(enc_weights, strict=False)
        if missing or unexpected:
            logging.debug(f"VQ-VAE load_state: missing={missing}, unexpected={unexpected}")

        self.vqvae.to(self.device).eval()
        self.transform_vqvqe = transforms.Lambda(scale_to_minus1_to_1)

    logging.info("VQ-VAE & codebook ready (lazy-initialized).")

    def build_index(self) -> None:
        """
        Build the Van Emde Boas (VEB) tree index for all slides.

        Steps:
          1. Reset in-memory accumulators.
          2. For each slide:
             - Load patch metadata and features.
             - Compute patch latent codes via VQ-VAE.
             - Map latents to integer indices.
             - Binarize texture features and record metadata.
          3. Construct the VEB tree over all keys.
          4. Save the VEB tree and metadata to disk.
        """

        self._ensure_vqvae_ready()

        # ---- reset accumulators ----
        self.meta.clear()
        self.keys.clear()
        logging.info("Reset metadata and key lists before index build.")

        # ---- iterate slides to populate keys + meta ----
        for slide_id, mosaic_pkl in self.paths.items():
            label = self.annotations.at[slide_id, 'category']
            patient_id = self.annotations.at[slide_id, 'patient']
            logging.debug(f"Indexing slide {slide_id}: category={label}, patient={patient_id}")

            # ---- load patches and features ----
            mosaic_data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=True)
            patches = mosaic_data['patches']

            # ---- compute latent codes via VQ-VAE ----
            with torch.no_grad():
                latents = compute_latent_features(
                    mosaic_pkl,
                    transform=self.transform_vqvqe,
                    vqvae=self.vqvae,
                    device=self.device,
                    batch_size=8,
                    num_workers=self.config['experiment']['num_workers'],
                )
                logging.debug(f"Computed {latents.shape[0]} latents for slide {slide_id}")

            # ---- map latents to integer indices ----
            slide_index = slide_to_index(
                latents,
                self.codebook_semantic,
                pool_layers=self.pool_layers,
                pool=self.pool
            )

            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
            logging.debug(f"Mapped latents to integer indices for slide {slide_id}")

            # ---- record binarized features and locations ----
            for idx, key in enumerate(slide_index):
                # 1) original numeric vector (e.g., float32[dim])
                vec  = patches[idx]['feature']

                # 2) threshold -> 0/1, then pack 8 bits per byte
                bits_packed = np.packbits((vec > 0).astype('uint8')).tobytes()

                x, y = patches[idx]['loc']
                entry = {
                     'slide_name': slide_id,
                     'bits'      : bits_packed,   
                     'patient_id': patient_id,
                     'category'  : label,
                     'x': x,
                     'y': y,}  
                self.meta.setdefault(key, []).append(entry)
                self.keys.append(int(key))

                # 3) free the large float array right now
                patches[idx]['feature'] = None
        
            gc.collect()
            del latents, mosaic_data, patches
            torch.cuda.empty_cache()   

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            log_mem(f"after slide {slide_id}")

        logging.info(f"Collected total of {len(self.keys)} keys from all slides.")

        # ---- build VEB tree ----
        universe = max(self.keys)
        logging.info(f"Universe size of VEB tree: {universe}")
        self.vebtree = VEB(universe)
        for k in self.keys:
            self.vebtree.insert(k)
        logging.info("VEB tree constructed successfully.")

        # ---- save data structures to disk ----
        with open(self.index_veb_path, 'wb') as f:
            pickle.dump(self.vebtree, f)
        with open(self.meta_database_path, 'wb') as f:
            pickle.dump(self.meta, f)
        logging.info(f"Saved VEB tree to {self.index_veb_path!r} and metadata to {self.meta_database_path!r}")
        
    def build_index_shards(self, resume: Optional[bool] = None, shard_size: Optional[int] = None) -> None:
        """
        Build the VEB index with resumable checkpoints.
        - Stores partial self.meta/self.keys to disk in shards every `shard_size` slides.
        - On resume, skips finished slides and continues.
        """

        self._ensure_vqvae_ready()

        if resume is None:
            resume = bool(self.resume_shards)   # from params/HYPERPARAMS
        if shard_size is None:
            shard_size = int(self.shard_size)   # from params/HYPERPARAMS
        
        # -------- paths --------
        project_dir = os.path.join("experiments", self.config['experiment']['project_name'])
        sish_dir    = os.path.join(project_dir, "sish")
        os.makedirs(sish_dir, exist_ok=True)

        shard_dir   = os.path.join(sish_dir, f"tmp_shards_{self.mosaic_string}")
        os.makedirs(shard_dir, exist_ok=True)

        self.index_veb_path      = os.path.join(sish_dir, f"veb_{self.mosaic_string}.pkl")
        self.meta_database_path  = os.path.join(sish_dir, f"meta_{self.mosaic_string}.pkl")
        manifest_path            = os.path.join(shard_dir, "manifest.json")

        # -------- load manifest or init --------
        if resume and os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                mani = json.load(f)
            processed_slides = set(mani["processed_slides"])
            shard_idx        = mani["next_shard_idx"]
            max_key_seen     = mani["max_key_seen"]
            logging.info(f"[SISH] Resuming. {len(processed_slides)} slides already done. Next shard {shard_idx}.")
        else:
            processed_slides = set()
            shard_idx        = 0
            max_key_seen     = -1

        # -------- scratch batches --------
        meta_batch  = defaultdict(list)   # key -> list[entry] (temporary)
        keys_batch  = []                  # list[int] (temporary)

        def flush_shard():
            nonlocal shard_idx, meta_batch, keys_batch
            if not keys_batch and not meta_batch:
                return
            meta_path = os.path.join(shard_dir, f"meta_shard_{shard_idx:04d}.pkl")
            keys_path = os.path.join(shard_dir, f"keys_shard_{shard_idx:04d}.npy")
            with open(meta_path, "wb") as f:
                pickle.dump(dict(meta_batch), f, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(keys_path, np.array(keys_batch, dtype=np.int64))
            logging.info(f"[SISH] Flushed shard {shard_idx:04d}: {len(keys_batch)} keys, {len(meta_batch)} unique keys")

            # reset
            meta_batch.clear()
            keys_batch.clear()
            shard_idx += 1
            gc.collect()

            # update manifest on disk
            mani = {
                "processed_slides": sorted(processed_slides),
                "next_shard_idx": shard_idx,
                "max_key_seen": max_key_seen
            }
            with open(manifest_path, "w") as f:
                json.dump(mani, f, indent=2)

        # -------- main loop --------
        to_iterate = list(self.paths.items())
        logging.info("Reset metadata and key lists before index build." if not processed_slides else
                    "Continuing metadata build.")

        for slide_id, mosaic_pkl in to_iterate:
            if slide_id in processed_slides:
                continue

            label      = self.annotations.at[slide_id, 'category']
            patient_id = self.annotations.at[slide_id, 'patient']
            logging.debug(f"Indexing slide {slide_id}: category={label}, patient={patient_id}")

            mosaic_data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=True)
            patches     = mosaic_data['patches']

            # latent calc
            with torch.no_grad():
                latents = compute_latent_features(
                    mosaic_pkl,
                    transform=self.transform_vqvqe,
                    vqvae=self.vqvae,
                    device=self.device,
                    batch_size=8,
                    num_workers=self.config['experiment']['num_workers'],
                )
            slide_index = slide_to_index(
                latents,
                self.codebook_semantic,
                pool_layers=self.pool_layers,
                pool=self.pool
            )

            for idx, key in enumerate(slide_index):
                vec  = patches[idx]['feature']
                bits_packed = np.packbits((vec > 0).astype('uint8')).tobytes()
                x, y = patches[idx]['loc']
                entry = {
                    'slide_name': slide_id,
                    'bits'      : bits_packed,
                    'patient_id': patient_id,
                    'category'  : label,
                    'x': x,
                    'y': y,
                }
                meta_batch[key].append(entry)
                keys_batch.append(int(key))
                max_key_seen = max(max_key_seen, int(key))
                patches[idx]['feature'] = None

            # housekeeping
            processed_slides.add(slide_id)
            del latents, mosaic_data, patches
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
            gc.collect()

            # flush every shard_size slides
            if len(processed_slides) % shard_size == 0:
                flush_shard()

        # final flush
        flush_shard()

        # -------- combine shards into final meta/keys (streaming) --------
        logging.info("[SISH] Loading shards to assemble final meta/keys for VEB build")
        self.meta = defaultdict(list)
        self.keys = []

        # stream merge to avoid big memory spikes
        for i in range(shard_idx):
            meta_path = os.path.join(shard_dir, f"meta_shard_{i:04d}.pkl")
            keys_path = os.path.join(shard_dir, f"keys_shard_{i:04d}.npy")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "rb") as f:
                part_meta = pickle.load(f)
            for k, lst in part_meta.items():
                self.meta[k].extend(lst)
            part_keys = np.load(keys_path)
            self.keys.extend(part_keys.tolist())
            # free
            del part_meta, part_keys
            gc.collect()

        logging.info(f"Collected total of {len(self.keys)} keys from all slides.")

        # -------- build VEB --------
        universe = max_key_seen
        logging.info(f"Universe size of VEB tree: {universe}")
        self.vebtree = VEB(universe)
        for k in self.keys:
            self.vebtree.insert(k)
        logging.info("VEB tree constructed successfully.")

        # save final objects
        with open(self.index_veb_path, 'wb') as f:
            pickle.dump(self.vebtree, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.meta_database_path, 'wb') as f:
            pickle.dump(dict(self.meta), f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Saved VEB tree to {self.index_veb_path!r} and metadata to {self.meta_database_path!r}")

    def leave_one_patient(self, patient_id: str) -> None:
        """
        Exclude all entries for a given patient from the metadata.

        Args:
            patient_id (str): Unique patient ID to leave out of the index.

        Returns:
            None: Updates self.meta_clean in-place.
        """
        # ---- choose metadata source based on mode ----
        if self.return_patch_matches:
            # In patch-level mode, do not filter anything
            self.meta_clean = self.meta
            logging.debug(f"No patient filtering applied (patch-level mode); kept all {len(self.meta)} keys")
        else:
            # In slide-level mode, remove any entry belonging to the held-out patient
            filtered = {}
            for key, entries in self.meta.items():
                # ---- filter entries for this key ----
                kept = [e for e in entries if e['patient_id'] != patient_id]
                filtered[key] = kept
            self.meta_clean = filtered
            logging.info(f"Filtered out patient {patient_id}; metadata now has {len(self.meta_clean)} keys")

    def search(
        self, 
        query_index: int, 
        dense_feat: str, 
        patient_id: str, 
    ) -> list:
        """
        Implements the bidirectional VEB-guided search from the SISH paper.

        Args:
            query_index (int): Integer index of the query latent code.
            dense_feat (str): Binary string representing quantized DenseNet features.
            patient_id (str): Patient ID to exclude from retrieval.

        Returns:
            list of tuples: Each tuple is either
                (query_index, match_index, global_dist, hamming_dist,
                 slide_name, category, patient_id, x, y)
            for slide-mode, or
                (query_index, match_index, global_dist, hamming_dist,
                 patch_name, category)
            for patch-mode.
        """
        #logging.info(f"Starting search for query_index={query_index}, patient_id={patient_id}")

        # ---- section: generate seed indices ----
        seed_index = []
        seed_index_pre = [int(query_index - m * self.seed_interval_c * 1e11) for m in range(self.seed_fanout_t)]
        seed_index_succ = [int(query_index + m * self.seed_interval_c * 1e11) for m in range(self.seed_fanout_t)]
        seed_index.extend(seed_index_pre)
        seed_index.extend(seed_index_succ)
        #logging.debug(f"Generated {len(seed_index)} seed indices (pre + succ)")

        # ---- section: prepare results container ----
        res = []
        visited = {}

        # ---- section: backward and forward traversal ----
        for index in seed_index:
            # ---- backward search ----
            pre_prev = index
            p_count = 0
            while p_count < self.pre_step:
                pre = self.vebtree.predecessor(pre_prev)
                if pre is None or pre in visited:
                    break

                candidates = self.meta.get(pre, [])
                # filter out same-patient entries
                candidates_clean = [e for e in candidates if e['patient_id'] != patient_id]
                if not candidates_clean:
                    #logging.info(f"No candidates found for index {pre}; skipping")
                    pre_prev = pre
                    continue
                #logging.info(f"Found {len(candidates_clean)} candidates for index {pre}")

                # ---- compute hamming distances ----
                if len(candidates_clean) > 1:
                    dists = [hamming_bytes(e['bits'], dense_feat) for e in candidates_clean]
                    min_idx = int(np.argmin(dists))
                    hamming_dist = dists[min_idx]
                else:
                    min_idx = 0
                    hamming_dist = hamming_bytes(candidates_clean[0]['bits'], dense_feat)

                # ---- accept candidate if within threshold ----
                if hamming_dist <= self.hamming_thr:
                    #logging.info(f"Accepted candidate with hamming distance {hamming_dist} <= {self.hamming_thr}")
                    entry = candidates_clean[min_idx]
                    visited[pre] = True
                    if not self.return_patch_matches:
                        #logging.info(f"Slide mode: {entry['slide_name']}")
                        res.append((
                            query_index,
                            pre,
                            abs(pre - query_index),
                            hamming_dist,
                            entry['slide_name'],
                            entry['category'],
                            entry['patient_id'],
                            entry['x'],
                            entry['y'],
                        ))
                    else:
                        res.append((
                            query_index,
                            pre,
                            abs(pre - query_index),
                            hamming_dist,
                            entry['patch_name'],
                            entry['category'],
                        ))
                #else:
                #    logging.info(f"Hamming distance {hamming_dist} exceeds threshold {self.hamming_thr}; skipping")

                p_count += 1
                pre_prev = pre

            # ---- forward search ----
            succ_prev = index
            s_count = 0
            while s_count < self.succ_step:
                succ = self.vebtree.successor(succ_prev)
                if succ is None or succ in visited:
                    break
                candidates = self.meta.get(succ, [])
                candidates_clean = [e for e in candidates if e['patient_id'] != patient_id]

                if not candidates_clean:
                    #logging.info(f"No candidates found for index {succ}; skipping")
                    succ_prev = succ
                    continue
                #logging.info(f"Found {len(candidates_clean)} candidates for index {succ}")

                if len(candidates_clean) > 1:
                    dists = [hamming_bytes(e['bits'], dense_feat) for e in candidates_clean]
                    min_idx = int(np.argmin(dists))
                    hamming_dist = dists[min_idx]
                else:
                    min_idx = 0
                    hamming_dist = hamming_bytes(candidates_clean[0]['bits'], dense_feat)

                if hamming_dist <= self.hamming_thr:
                    #logging.info(f"Accepted candidate with hamming distance {hamming_dist} <= {self.hamming_thr}")
                    entry = candidates_clean[min_idx]
                    visited[succ] = True
                    if not self.return_patch_matches:
                        #logging.info(f"Accepted candidate: {entry}")
                        res.append((
                            query_index,
                            succ,
                            abs(succ - query_index),
                            hamming_dist,
                            entry['slide_name'],
                            entry['category'],
                            entry['patient_id'],
                            entry['x'],
                            entry['y'],
                        ))
                    else:
                        res.append((
                            query_index,
                            succ,
                            abs(succ - query_index),
                            hamming_dist,
                            entry['patch_name'],
                            entry['category'],
                        ))
                #else:
                    #logging.info(f"Hamming distance {hamming_dist} exceeds threshold {self.hamming_thr}; skipping")
                    
                s_count += 1
                succ_prev = succ

        #logging.warning(f"Search completed: found {len(res)} candidate(s)")

        return res

    """
    def preprocessing(self, latent: np.ndarray) -> np.ndarray:
        #logging.info("Running preprocessing to convert latent code(s) to index")

        # ---- compute index via internal helper ----
        mosaic_index = self._slide_to_index(latent)

        # ---- log result shape ----
        # if mosaic_index is an array, log its length; else log the single value
        try:
            length = len(mosaic_index)
        except TypeError:
            length = 1
        #logging.debug(f"Preprocessing produced {length} index value(s)")
        return mosaic_index
    """

    def postprocessing(self, res_tmp: list) -> list:
        """
        Sort raw search tuples by Hamming distance and convert into dicts.

        Args:
            res_tmp (list): List of raw tuples from `search()`, each containing:
                - query index
                - match index
                - global distance
                - hamming distance
                - slide_name/patch_name
                - category
                - [patient_id, x, y] if slide mode

        Returns:
            list of dict: Each dict maps field names to tuple values, sorted by hamming distance.
        """
        #logging.warning(f"Postprocessing {len(res_tmp)} raw search result(s)")

        # ---- sort by hamming distance (4th element) ----
        res_srt = sorted(res_tmp, key=lambda x: x[3])

        # ---- choose field names based on mode ----
        if self.return_patch_matches:
            field_names = ['query', 'index', 'global_dist', 'hamming_dist', 'patch_name', 'category']
        else:
            field_names = ['query', 'index', 'global_dist', 'hamming_dist',
                           'slide_name', 'category', 'patient_id', 'x', 'y']
        # ---- build list of dicts ----
        res_srt_dict = [dict(zip(field_names, tup)) for tup in res_srt]
        #logging.warning(f"Postprocessing returned {len(res_srt_dict)} formatted entries")
        return res_srt_dict

    def query(
        self,
        index: int,
        dense_feat: str,
        patient_id: str,
    ) -> list:
        """
        Perform a single leave-one-patient-out query.

        Args:
            index (int): Integer index of the query latent code.
            dense_feat (str): Binarized DenseNet feature string for the query.
            patient_id (str): Patient ID to exclude during search.

        Returns:
            list of dict: Top-k retrieval results, each dict containing:
                - query_slide_id
                - query_label
                - predicted_label
                - top_k: list of {slide_id, label, distance}
        """
        #logging.info(f"Querying index={index} (patient_id={patient_id}) with topk={self.k}")

        # ---- run raw search ----
        indices_nn = self.search(
            index,
            dense_feat,
            patient_id,
        )
        #logging.info(f"Raw search returned {len(indices_nn)} entries")

        # ---- format via postprocessing ----
        results = self.postprocessing(indices_nn)
        #logging.info(f"Postprocessed to {len(results)} formatted entries")

        return results
    
    def compute_database_weights(self, patient_id):
        """
        Compute inverse-frequency weights based on _indexed_ slides in self.meta,
        excluding the held-out patient.

        Returns:
            dict[label, float]: weight for each label.
        """
        # Count searchable slides per label, excluding the query‐patient
        categories = self.annotations['category'].unique()
        total_per_label = {cat: 0 for cat in categories}
        for patch_bag in self.meta.values():
            for entry in patch_bag:
                if entry['patient_id'] == patient_id:
                    continue
                total_per_label[entry['category']] += 1

        # Drop zero‐count labels
        total_per_label = {lbl: cnt for lbl, cnt in total_per_label.items() if cnt > 0}

        # Build inverse‐frequency weights
        inv_sum = sum(1.0 / cnt for cnt in total_per_label.values())
        if inv_sum == 0:
            # fallback to uniform if nothing left
            return {lbl: 1.0 for lbl in total_per_label}

        norm_fact = self.k / inv_sum
        weights = {lbl: norm_fact * (1.0 / cnt) for lbl, cnt in total_per_label.items()}

        logging.debug(f"Database counts per label: {total_per_label}")
        logging.debug(f"Computed database weights:    {weights}")
        return weights

    def clean_single_result(self, slide_id, data: dict, weights) -> dict:
        """
        Clean & aggregate retrieval outputs for one query slide.

        Returns a single dict with:
            - query_slide_id
            - query_label
            - predicted_label
            - top_k: list of {slide_id, label, distance}
        """
        bags = data['results']
        total_bags = sum(len(b) for b in bags)
        if total_bags == 0:
            #logging.warning(f"No retrievals for slide {slide_id}; skipping")
            return {
                "query_slide_id": slide_id,
                "query_label": data['label_query'],
                "predicted_label": None,
                "top_k": []
            }

        # 1) compute uncertainties
        bag_summary = []
        label_count_summary = {}

        for idx, bag in enumerate(bags):
            # bag is a list of dicts from self.query(...), each has 'hamming_dist'
            hams = [entry['hamming_dist'] for entry in bag]
            hams.sort()

            ent, label_count, _ = Uncertainty_Cal(bag, weights)

            # Skip empty bags if your Clean() can’t handle them
            if ent is not None and hams:
                label_count_summary[idx] = label_count
                # (bag_index, entropy, list_of_hamming_distances, bag_len)
                bag_summary.append((idx, ent, hams, len(hams)))

        # 2) Hamming‐based clean & prediction filtering
        lengths = [b[3] for b in bag_summary]
        bag_summary, hamming_thr = Clean(lengths, bag_summary)
        removed = Filtered_BY_Prediction(bag_summary, label_count_summary)

        # 3) assemble top‐k
        ret_final = []
        visited = set()
        for bag_idx, unc, _, _ in bag_summary:
            for entry in bags[bag_idx]:
                sid = entry['slide_name']
                hd  = entry['hamming_dist']
                lbl = entry.get('diagnosis', entry.get('category'))
                if unc == 0 or (hd <= hamming_thr and sid not in visited):
                    ret_final.append((sid, hd, lbl, unc, bag_idx))
                    visited.add(sid)

        # sort & truncate
        ret_final = [e for e in sorted(ret_final, key=lambda x: (x[3], x[1]))
                     if e[-1] not in removed
                     ]
        logging.info(f"[SISH] Retrieved {len(ret_final)} candidates after cleaning for slide {slide_id}")

        top_k_info = [{"slide_id": sid, "label": lbl, "distance": float(hd)}
                      for (sid, hd, lbl, _, _) in ret_final[: self.k]
                      ]
        
        predicted = (Counter([d["label"] for d in top_k_info])
                    .most_common(1)[0][0] if top_k_info else None)

        return {
            "query_slide_id": slide_id,
            "query_label":    data['label_query'],
            "predicted_label": predicted,
            "top_k":          top_k_info
        }

    def leave_one_patient_out(self) -> list:
        """
        Leave‐one‐patient‐out retrieval benchmark.
        """
        logging.info(f"Looking for index files, VEB tree at {self.index_veb_path} and metadata at {self.meta_database_path} ")
        if os.path.exists(self.index_veb_path) and os.path.exists(self.meta_database_path):
            # both files are there -> load them
            with open(self.index_veb_path, 'rb') as f:
                self.vebtree = pickle.load(f)
            with open(self.meta_database_path, 'rb') as f:
                self.meta = pickle.load(f)
            logging.info(f"Loaded VEB tree from {self.index_veb_path!r} and metadata from {self.meta_database_path!r}")
        else:
            # files missing or incomplete -> rebuild
            logging.warning("Index files not found or incomplete; rebuilding index.")
            #self.build_index()
            self.build_index_shards()

        topk_results = []
        for slide_id in self.paths:
            patient_id = self.annotations.at[slide_id, 'patient']
            label      = self.annotations.at[slide_id, 'category']

            # gather all patch‐bags for this query slide
            patient_indexes = []
            for key, entries in self.meta.items():
                for entry in entries:
                    if entry['slide_name'] == slide_id:
                        patient_indexes.append((key, entry['bits']))

            # run the per‐patch query
            slide_outputs = [self.query(idx, feat, patient_id) for idx, feat in patient_indexes]
            logging.info(f"{len(slide_outputs)} retrieved slides for slide {slide_id}")

            # compute weights excluding this patient
            weights = self.compute_database_weights(patient_id)

            # clean up into one dict and append
            result = {
                'results':     slide_outputs,
                'label_query': label
            }
            cleaned = self.clean_single_result(slide_id, result, weights)
            topk_results.append(cleaned)

        logging.info(f"LOPO done: {len(topk_results)} slides")
        return topk_results


"""def clean_results(self, results: dict) -> list:
 
        Aggregate and clean retrieval outputs across all query slides using uncertainty filtering.

        Args:
            results (dict): Mapping from slide_id to a dict with:
                - 'results': list of per-patch query outputs
                - 'label_query': ground-truth label for the slide

        Returns:
            list of dict: Final cleaned retrieval entries, each containing:
                - query_slide_id
                - query_label
                - predicted_label
                - top_k: list of {slide_id, label, distance}
 
        logging.info(f"Cleaning aggregated results for {len(results)} slides")
        topk_results = []

        # ---- count queries per label ----
        categories = self.annotations['category'].unique()
        total_per_label = {cat: 0 for cat in categories}
        for slide_id, data in results.items():
            total_per_label[data['label_query']] += 1
        logging.debug(f"Total queries per label: {total_per_label}")

        # ---- compute weights (inverse-frequency) ----
        inv_sum = sum(1.0 / cnt for cnt in total_per_label.values() if cnt > 0)
        norm_fact = self.k / inv_sum
        weight = {label: norm_fact * (1.0 / cnt) for label, cnt in total_per_label.items()}
        logging.debug(f"Computed label weights: {weight}")

        # ---- process each label group ----
        for eval_label, w in weight.items():
            for slide_id, data in results.items():
                if data['label_query'] != eval_label:
                    continue

                bags = data['results']
                # skip if no retrievals at all #TODO: still needs to be in the results
                total_bags = sum(len(b) for b in bags)
                if total_bags == 0:
                    topk_results.append({
                        "query_slide_id": slide_id,
                        "query_label": data['label_query'],
                        "predicted_label": None,
                        "top_k": []
                    })
                    logging.warning(f"No retrievals for slide {slide_id}; skipping")
                    continue

                # ---- compute uncertainty and summary per bag ----
                bag_summary = []
                label_count_summary = {}
                for idx, bag in enumerate(bags):
                    ent, label_count, _ = Uncertainty_Cal(bag, weight)
                    if ent is not None:
                        label_count_summary[idx] = label_count
                        bag_summary.append((idx, ent, None, len(bag)))  # distance unused here
                logging.debug(f"Slide {slide_id} bag_summary: {bag_summary}")

                # ---- clean by Hamming threshold and prediction filtering ----
                lengths = [b[3] for b in bag_summary]
                bag_summary, hamming_thr = Clean(lengths, bag_summary)
                removed = Filtered_BY_Prediction(bag_summary, label_count_summary)
                logging.debug(f"After Clean & Filter: hamming_thr={hamming_thr}, removed={removed}")

                # ---- assemble final top-k per slide ----
                retrieval_final = []
                visited = set()
                for b in bag_summary:
                    bag_idx, uncertainty, _, _ = b
                    for entry in bags[bag_idx]:
                        sid = entry['slide_name']
                        hd = entry['hamming_dist']
                        lbl = entry.get('diagnosis', entry.get('category'))
                        if uncertainty == 0 or (hd <= hamming_thr and sid not in visited):
                            retrieval_final.append((sid, hd, lbl))
                            visited.add(sid)

                # ---- sort, filter removed, limit to self.k ----
                retrieved = sorted(retrieval_final, key=lambda x: x[1])[:self.k]
                top_k_info = [
                    {"slide_id": sid, "label": lbl, "distance": float(dist)}
                    for sid, dist, lbl in retrieved
                ]
                predicted = Counter([d["label"] for d in top_k_info]).most_common(1)[0][0] if top_k_info else None

                topk_results.append({
                    "query_slide_id": slide_id,
                    "query_label": data['label_query'],
                    "predicted_label": predicted,
                    "top_k": top_k_info
                })

        logging.info(f"Completed cleaning: produced {len(topk_results)} final entries")
        return topk_results"""