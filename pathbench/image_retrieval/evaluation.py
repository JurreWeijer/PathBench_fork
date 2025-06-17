import logging
from collections import defaultdict, Counter
import re
import numpy as np
from typing import Dict, List 

def parse_metric_names(metric_names):
    """
    Parse metric strings like 'hit_at_5', 'mmv_at_10', or 'map_at_3' into a structured format.

    Extracts the metric type (e.g., "hit", "mmv", "map") and the integer value of k, returning
    a dictionary that maps each metric type to a set of its corresponding k values.

    Args:
        metric_names (list): List of metric strings, expected format '<metric>_at_<k>'.

    Returns:
        dict: Dictionary mapping metric types to sets of integer k values.
              Example: {'hit': {5, 10}, 'map': {3}}
    """
    parsed = defaultdict(set)

    for name in metric_names:
        # Match pattern like "hit_at_5" or "map_at_10"
        match = re.match(r"(hit|mmv|map)_at_(\d+)", name)
        if match:
            metric, k = match.groups()
            parsed[metric].add(int(k))

    return parsed

def compute_hit_at_k(results, k, count_short_as_miss=True):
    hit_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for result in results:
        lbl = result['query_label']

        top = [slide['label'] for slide in result.get('top_k', [])][:k]
        if len(top) < k and not count_short_as_miss:
            continue
        
        total_counts[lbl] += 1
            
        # count_short_as_miss=True => we treat len(top)<k as a miss (no-op)
        if lbl in top:
            hit_counts[lbl] += 1

    return {l: (hit_counts[l] / total_counts[l] if total_counts[l] else 0.0)
            for l in total_counts}

def compute_mmv_at_k(
    results: list,
    k: int,
    count_short_as_miss: bool = True
) -> dict:
    """
    Computes Majority Vote@K for each label, with optional counting of 
    'too few results' as misses.

    Args:
        results: List of dicts with 'query_label' and 'top_k' keys.
        k: Number of top results to consider.
        count_short_as_miss: 
            - True: if len(top_k) < k, still counts as one query (but a miss).
            - False: skip any query with fewer than k results entirely.

    Returns:
        Dict[label, float]: mMV@K per label.
    """
    correct_counts = defaultdict(int)
    total_counts   = defaultdict(int)

    for r in results:
        lbl = r['query_label']
        top = [s['label'] for s in r.get('top_k', [])][:k]

        # handle too-short cases
        if len(top) < k:
            if not count_short_as_miss:
                continue
            # else: count this query but it can never be correct
        total_counts[lbl] += 1

        if top:
            majority = Counter(top).most_common(1)[0][0]
            if majority == lbl:
                correct_counts[lbl] += 1

    return {
        lbl: (correct_counts[lbl] / total_counts[lbl] if total_counts[lbl] else 0.0)
        for lbl in total_counts
    }


def compute_map_at_k(
    results: list,
    k: int,
    count_short_as_miss: bool = True
) -> dict:
    """
    Computes Mean Average Precision@K for each label, with optional counting of
    'too few results' as zero-AP.

    Args:
        results: List of dicts with 'query_label' and 'top_k' keys.
        k: Number of top results to consider.
        count_short_as_miss: 
            - True: if len(top_k) < k, treat AP=0 for that query.
            - False: skip query entirely.

    Returns:
        Dict[label, float]: mAP@K per label.
    """
    ap_per_label = defaultdict(list)

    for r in results:
        lbl = r['query_label']
        top = [s['label'] for s in r.get('top_k', [])][:k]

        if len(top) < k:
            if not count_short_as_miss:
                continue
            # else: AP=0
            ap_per_label[lbl].append(0.0)
            continue

        num_rel = 0
        precisions = []
        for i, lab in enumerate(top):
            if lab == lbl:
                num_rel += 1
                precisions.append(num_rel / (i + 1))

        ap = float(np.mean(precisions)) if precisions else 0.0
        ap_per_label[lbl].append(ap)

    return {
        lbl: (float(np.mean(aps)) if aps else 0.0)
        for lbl, aps in ap_per_label.items()
    }


def compute_micro_average(
    results: list,
    mtype: str,
    k: int,
    count_short_as_miss: bool = True
) -> float:
    """
    Compute the micro-average for one metric type at K.

    Args:
        results: List of dicts with 'query_label' and 'top_k'.
        mtype: One of 'hit', 'mmv', 'map'.
        k: Number of top results to consider.
        count_short_as_miss: controls skip vs count-as-miss.

    Returns:
        float: The micro-averaged score.
    """
    # filter or not
    if count_short_as_miss:
        valid = results
    else:
        valid = [r for r in results if len(r.get('top_k', [])) >= k]

    nq = len(valid)
    if nq == 0:
        return 0.0

    if mtype == 'hit':
        hits = 0
        for r in valid:
            top = [s['label'] for s in r.get('top_k', [])][:k]
            if len(top) < k and not count_short_as_miss:
                continue
            if r['query_label'] in top:
                hits += 1
        return hits / nq

    if mtype == 'mmv':
        correct = 0
        for r in valid:
            top = [s['label'] for s in r.get('top_k', [])][:k]
            if len(top) < k and not count_short_as_miss:
                continue
            if top:
                maj = Counter(top).most_common(1)[0][0]
                if maj == r['query_label']:
                    correct += 1
        return correct / nq

    if mtype == 'map':
        ap_list = []
        for r in valid:
            top = [s['label'] for s in r.get('top_k', [])][:k]
            if len(top) < k and not count_short_as_miss:
                continue
            num_rel = 0
            precisions = []
            for i, lab in enumerate(top):
                if lab == r['query_label']:
                    num_rel += 1
                    precisions.append(num_rel / (i + 1))
            ap_list.append(float(np.mean(precisions)) if precisions else 0.0)
        return float(np.mean(ap_list)) if ap_list else 0.0

    # unknown type
    return 0.0

def evaluate_retrieval_metrics(
    results: List[Dict],
    metric_names: List[str],
    count_short_as_miss: bool = True
) -> Dict:
    """
    Computes retrieval metrics and their macro/micro averages.

    Args:
        results: list of dicts with 'query_label' and 'top_k'
        metric_names: e.g. ['hit_at_5','mmv_at_3','map_at_5']
        count_short_as_miss: 
            - True: queries with fewer than k results are treated as misses (hit=0, etc.)
            - False: those queries are simply skipped
    
    Returns:
        dict of per-label and aggregate metrics
    """
    parsed = parse_metric_names(metric_names)
    metrics = {}

    for mtype, ks in parsed.items():
        for k in ks:
            # decide which queries to include
            if count_short_as_miss:
                valid = results
            else:
                valid = [r for r in results if len(r.get('top_k', [])) >= k]
            
            if not valid:
                logging.warning(f"No valid queries for {mtype}@{k} (count_short_as_miss={count_short_as_miss})")
                continue

            # dispatch to the right metric calculator, passing along the flag
            if mtype == 'hit':
                per_class = compute_hit_at_k(valid, k, count_short_as_miss)
            elif mtype == 'mmv':
                per_class = compute_mmv_at_k(valid, k, count_short_as_miss)
            elif mtype == 'map':
                per_class = compute_map_at_k(valid, k, count_short_as_miss)
            else:
                logging.warning(f"Unknown metric type: {mtype}")
                continue

            key = f"{mtype}_at_{k}"
            metrics[key] = per_class

            # macro average
            macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
            metrics[f"{key}_macro"] = {'all': macro}

            # micro average
            micro = compute_micro_average(valid, mtype, k, count_short_as_miss)
            metrics[f"{key}_micro"] = {'all': micro}

    return metrics