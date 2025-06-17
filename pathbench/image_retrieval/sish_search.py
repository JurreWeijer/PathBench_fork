import os
import time
import numpy as np
import h5py
import torch
import openslide
import copy
import pickle
import pandas as pd
from collections import OrderedDict
from torchvision.models import densenet121
from torchvision import transforms
from tqdm import tqdm
from typing import Dict, Optional
import slideflow as sf
import multiprocessing
import logging
import multiprocessing as mp
from collections import Counter

from ..image_retrieval.sish_index import min_max_binarized, compute_latent_features, slide_to_index
from ..image_retrieval.sish_eval import Uncertainty_Cal, Clean, Filtered_BY_Prediction
from ..image_retrieval.sish_vqvae import LargeVectorQuantizedVAE_Encode
from ..image_retrieval.sish_veb import VEB
from ..image_retrieval.utils import load_patch_dicts_pickle

logger = logging.getLogger(__name__)

class SISHDatabase:
    """
    SISHDatabase implements the Selection of Informative Samples in Histopathology (SISH) retrieval pipeline.
    It builds and manages an index over slide-level patch mosaics using a VQ-VAE encoder,
    hierarchical pooling, and a Van Emde Boas (VEB) tree for efficient nearest-neighbor search.
    """
    def __init__(
        self,
        config: dict,
        slide_mosaic_paths: Dict[str, str],
        k: int,
        mosaic_string: str
    ) -> None:
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
        # ---- store parameters ----
        self.config = config
        self.slide_mosaics = slide_mosaic_paths
        self.topk = k
        self.mosaic_string = mosaic_string

        # ---- initialize in-memory structures ----
        self.meta = {}   # key:int -> List[meta-dict]
        self.keys = []   # flat list of all keys for VEB
        self.vebtree = None
        self.is_patch = False

        # ---- create directories for index storage ----
        project_dir = os.path.join("experiments", config['experiment']['project_name'])

        sish_dir = os.path.join(project_dir, "sish")
        os.makedirs(sish_dir, exist_ok=True)

        self.index_veb_path = os.path.join(sish_dir, f"veb_{mosaic_string}.pkl")
        self.meta_database_path = os.path.join(sish_dir, f"meta_{mosaic_string}.pkl")

        # ---- load slide-level annotations ----
        annotations_path = config['experiment']['annotation_file']
        self.annotations = pd.read_csv(annotations_path).set_index('slide')
        logging.info(f"Loaded annotations for {len(self.annotations)} slides from {annotations_path}")

        # ---- initialize VQ-VAE encoder and codebook ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        codebook_path = config["SISH_metrics"]["codebook_semantic"]
        checkpoint_path = config["SISH_metrics"]["vqvae_checkpoint"]
        self.codebook_semantic = torch.load(codebook_path)
        logging.debug(f"Loaded semantic codebook from {codebook_path}")

        # ---- instantiate and load encode-only VQ-VAE ----
        self.vqvae = LargeVectorQuantizedVAE_Encode(
            code_dim=256,
            code_size=128
        )
        # transform: image tensor [0,1] -> [-1,1]
        self.transform_vqvqe = transforms.Lambda(lambda x: 2 * x - 1)

        # ---- load encoder & codebook weights, stripping 'module.' prefix ----
        raw_checkpoint = torch.load(checkpoint_path)['model']
        enc_weights = OrderedDict({
            k[len("module."):]: v
            for k, v in raw_checkpoint.items()
            if k.startswith("module.encoder.") or k.startswith("module.codebook.")
        })

        self.vqvae.load_state_dict(enc_weights)
        self.vqvae.to(self.device).eval()
        logging.info("VQ-VAE encoder weights loaded; model set to eval mode.")

        # ---- setup pooling layers for hierarchical sums ----
        self.pool_layers = [
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.AvgPool2d((2, 2))
        ]

        # ---- create multiprocessing pool ----
        num_workers = mp.cpu_count()
        self.pool = mp.Pool(num_workers)
        logging.debug(f"Initialized multiprocessing pool with {num_workers} workers")

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
        # ---- reset accumulators ----
        self.meta.clear()
        self.keys.clear()
        logging.info("Reset metadata and key lists before index build.")

        # ---- iterate slides to populate keys + meta ----
        for slide_id, mosaic_pkl in self.slide_mosaics.items():
            label = self.annotations.at[slide_id, 'category']
            patient_id = self.annotations.at[slide_id, 'patient']
            logging.debug(f"Indexing slide {slide_id}: category={label}, patient={patient_id}")

            # ---- load patches and features ----
            mosaic_data = load_patch_dicts_pickle(mosaic_pkl, reconstruct_features=True)
            patches = mosaic_data['patches']

            # ---- compute latent codes via VQ-VAE ----
            latents = compute_latent_features(
                mosaic_pkl,
                transform=self.transform_vqvqe,
                vqvae=self.vqvae,
                device=self.device,
                batch_size=16,
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
            logging.debug(f"Mapped latents to integer indices for slide {slide_id}")

            # ---- record binarized features and locations ----
            for idx, key in enumerate(slide_index):
                bin_feat = min_max_binarized(patches[idx]['feature'])
                x, y = patches[idx]['loc']
                entry = {
                    'slide_name': slide_id,
                    'dense_binarized': bin_feat,
                    'patient_id': patient_id,
                    'category': label,
                    'x': x,
                    'y': y,
                }
                self.meta.setdefault(key, []).append(entry)
                self.keys.append(int(key))

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

    def leave_one_patient(self, patient_id: str) -> None:
        """
        Exclude all entries for a given patient from the metadata.

        Args:
            patient_id (str): Unique patient ID to leave out of the index.

        Returns:
            None: Updates self.meta_clean in-place.
        """
        # ---- choose metadata source based on mode ----
        if self.is_patch:
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
        pre_step: int, 
        succ_step: int, 
        C: int, 
        T: int, 
        thrsh: int
    ) -> list:
        """
        Implements the bidirectional VEB-guided search from the SISH paper.

        Args:
            query_index (int): Integer index of the query latent code.
            dense_feat (str): Binary string representing quantized DenseNet features.
            patient_id (str): Patient ID to exclude from retrieval.
            pre_step (int): Number of predecessors to traverse in backward search.
            succ_step (int): Number of successors to traverse in forward search.
            C (int): Interval width factor for seed index expansion.
            T (int): Number of times to expand the seed index on each side.
            thrsh (int): Hamming-distance threshold for accepting candidates.

        Returns:
            list of tuples: Each tuple is either
                (query_index, match_index, global_dist, hamming_dist,
                 slide_name, category, patient_id, x, y)
            for slide-mode, or
                (query_index, match_index, global_dist, hamming_dist,
                 patch_name, category)
            for patch-mode.
        """
        logging.info(f"Starting search for query_index={query_index}, patient_id={patient_id}")

        # ---- section: generate seed indices ----
        seed_index = []
        seed_index_pre = [int(query_index - m * C * 1e11) for m in range(T)]
        seed_index_succ = [int(query_index + m * C * 1e11) for m in range(T)]
        seed_index.extend(seed_index_pre)
        seed_index.extend(seed_index_succ)
        logging.debug(f"Generated {len(seed_index)} seed indices (pre + succ)")

        # ---- section: prepare results container ----
        res = []
        visited = {}

        # ---- section: backward and forward traversal ----
        for index in seed_index:
            # ---- backward search ----
            pre_prev = index
            p_count = 0
            while p_count < pre_step:
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
                    dists = [
                        bin(int(e['dense_binarized'], 2) ^ int(dense_feat, 2)).count('1')
                        for e in candidates_clean
                    ]
                    min_idx = int(np.argmin(dists))
                    hamming_dist = dists[min_idx]
                else:
                    min_idx = 0
                    hamming_dist = bin(
                        int(candidates_clean[0]['dense_binarized'], 2) ^ int(dense_feat, 2)
                    ).count('1')

                # ---- accept candidate if within threshold ----
                if hamming_dist <= thrsh:
                    #logging.info(f"Accepted candidate with hamming distance {hamming_dist} <= {thrsh}")
                    entry = candidates_clean[min_idx]
                    visited[pre] = True
                    if not self.is_patch:
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
                    #logging.info(f"Hamming distance {hamming_dist} exceeds threshold {thrsh}; skipping")

                p_count += 1
                pre_prev = pre

            # ---- forward search ----
            succ_prev = index
            s_count = 0
            while s_count < succ_step:
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
                    dists = [
                        bin(int(e['dense_binarized'], 2) ^ int(dense_feat, 2)).count('1')
                        for e in candidates_clean
                    ]
                    min_idx = int(np.argmin(dists))
                    hamming_dist = dists[min_idx]
                else:
                    min_idx = 0
                    hamming_dist = bin(
                        int(candidates_clean[0]['dense_binarized'], 2) ^ int(dense_feat, 2)
                    ).count('1')

                if hamming_dist <= thrsh:
                    #logging.info(f"Accepted candidate with hamming distance {hamming_dist} <= {thrsh}")
                    entry = candidates_clean[min_idx]
                    visited[succ] = True
                    if not self.is_patch:
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
                    #logging.info(f"Hamming distance {hamming_dist} exceeds threshold {thrsh}; skipping")
                    
                s_count += 1
                succ_prev = succ

        logging.warning(f"Search completed: found {len(res)} candidate(s)")

        return res

    def preprocessing(self, latent: np.ndarray) -> np.ndarray:
        """
        Convert VQ-VAE latent code into integer index (or indices).

        Args:
            latent (np.ndarray): Latent map(s) from the VQ-VAE encoder, shape (H, W) or (N, H, W).

        Returns:
            np.ndarray: Integer index (or array of indices) representing each latent map.
        """
        logging.info("Running preprocessing to convert latent code(s) to index")

        # ---- compute index via internal helper ----
        mosaic_index = self._slide_to_index(latent)

        # ---- log result shape ----
        # if mosaic_index is an array, log its length; else log the single value
        try:
            length = len(mosaic_index)
        except TypeError:
            length = 1
        logging.debug(f"Preprocessing produced {length} index value(s)")
        return mosaic_index

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
        logging.warning(f"Postprocessing {len(res_tmp)} raw search result(s)")

        # ---- sort by hamming distance (4th element) ----
        res_srt = sorted(res_tmp, key=lambda x: x[3])

        # ---- choose field names based on mode ----
        if self.is_patch:
            field_names = ['query', 'index', 'global_dist', 'hamming_dist', 'patch_name', 'category']
        else:
            field_names = ['query', 'index', 'global_dist', 'hamming_dist',
                           'slide_name', 'category', 'patient_id', 'x', 'y']
        # ---- build list of dicts ----
        res_srt_dict = [dict(zip(field_names, tup)) for tup in res_srt]
        logging.warning(f"Postprocessing returned {len(res_srt_dict)} formatted entries")
        return res_srt_dict

    def query(
        self,
        index: int,
        dense_feat: str,
        patient_id: str,
        pre_step: int = 375,
        succ_step: int = 375,
        C: int = 50,
        T: int = 10,
        thrsh: int = 512
    ) -> list:
        """
        Perform a single leave-one-patient-out query.

        Args:
            index (int): Integer index of the query latent code.
            dense_feat (str): Binarized DenseNet feature string for the query.
            patient_id (str): Patient ID to exclude during search.
            pre_step (int): Number of backward VEB steps (default 375).
            succ_step (int): Number of forward VEB steps (default 375).
            C (int): Interval width factor for seed expansion (default 50).
            T (int): Number of seed expansions on each side (default 10).
            thrsh (int): Hamming distance threshold for candidate acceptance (default 128).

        Returns:
            list of dict: Top-k retrieval results, each dict containing:
                - query_slide_id
                - query_label
                - predicted_label
                - top_k: list of {slide_id, label, distance}
        """
        logging.info(f"Querying index={index} (patient_id={patient_id}) with topk={self.topk}")

        # ---- run raw search ----
        indices_nn = self.search(
            index,
            dense_feat,
            patient_id,
            pre_step=pre_step,
            succ_step=succ_step,
            C=C,
            T=T,
            thrsh=thrsh
        )
        logging.warning(f"Raw search returned {len(indices_nn)} entries")

        # ---- format via postprocessing ----
        results = self.postprocessing(indices_nn)
        logging.warning(f"Postprocessed to {len(results)} formatted entries")

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

        norm_fact = self.topk / inv_sum
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
            logging.warning(f"No retrievals for slide {slide_id}; skipping")
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
            ent, label_count, _ = Uncertainty_Cal(bag, weights)
            if ent is not None:
                label_count_summary[idx] = label_count
                bag_summary.append((idx, ent, None, len(bag)))

        # 2) Hamming‐based clean & prediction filtering
        lengths = [b[3] for b in bag_summary]
        bag_summary, hamming_thr = Clean(lengths, bag_summary)
        removed = Filtered_BY_Prediction(bag_summary, label_count_summary)

        # 3) assemble top‐k
        retrieval_final = []
        visited = set()
        for bag_idx, unc, _, _ in bag_summary:
            for entry in bags[bag_idx]:
                sid = entry['slide_name']
                hd  = entry['hamming_dist']
                lbl = entry.get('diagnosis', entry.get('category'))
                if unc == 0 or (hd <= hamming_thr and sid not in visited):
                    retrieval_final.append((sid, hd, lbl))
                    visited.add(sid)

        # sort & truncate
        retrieved = sorted(retrieval_final, key=lambda x: x[1])[:self.topk]
        top_k_info = [
            {"slide_id": sid, "label": lbl, "distance": float(dist)}
            for sid, dist, lbl in retrieved
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
        # load or build index/meta…
        topk_results = []
        for slide_id in self.slide_mosaics:
            patient_id = self.annotations.at[slide_id, 'patient']
            label      = self.annotations.at[slide_id, 'category']

            # gather all patch‐bags for this query slide
            patient_indexes = []
            for key, entries in self.meta.items():
                for entry in entries:
                    if entry['slide_name'] == slide_id:
                        patient_indexes.append((key, entry['dense_binarized']))

            # run the per‐patch query
            slide_outputs = [self.query(idx, feat, patient_id) for idx, feat in patient_indexes]
            logging.warning(f"{len(slide_outputs)} number of retrieval for slide {slide_id}")

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
        norm_fact = self.topk / inv_sum
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

                # ---- sort, filter removed, limit to self.topk ----
                retrieved = sorted(retrieval_final, key=lambda x: x[1])[:self.topk]
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