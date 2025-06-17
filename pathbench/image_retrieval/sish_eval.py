import pickle
import time
import numpy as np
import operator
import argparse
import copy
import os
import math
import glob
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

def Uncertainty_Cal(bag, weight, is_organ=False):
    """
    Implementation of Weighted-Uncertainty-Cal in the paper.
    Input:
        bag (list): A list of dictionary which contain the searhc results for each mosaic
    Output:
        ent (float): The entropy of the mosaic retrieval results
        label_count (dict): The diagnois and the corresponding weight for each mosaic
        hamming_dist (list): A list of hamming distance between the input mosaic and the result
    """
    if len(bag) >= 1:
        label = []
        hamming_dist = []
        label_count = defaultdict(float)
        for bres in bag:
            label.append(bres['category'])
            hamming_dist.append(bres['hamming_dist'])

        # Counting the diagnoiss by weigted count
        # If the count is less than 1, round to 1
        for lb_idx, lb in enumerate(label):
            label_count[lb] += (1. / (lb_idx + 1)) * weight[lb]
        for k, v in label_count.items():
            if v < 1.0:
                v = 1.0
            else:
                label_count[k] = v

        # Normalizing the count to [0,1] for entropy calculation
        total = 0
        ent = 0
        for v in label_count.values():
            total += v
        for k in label_count.keys():
            label_count[k] = label_count[k] / total
        for v in label_count.values():
            ent += (-v * np.log2(v))
        return ent, label_count, hamming_dist
    else:
        return None, None, None


def Clean(len_info, bag_summary):
    """
    Implementation of Clean in the paper
    Input:
        len_info (list): The length of retrieval results for each mosaic
        bag_summary (list): A list that contains the positional index of mosaic,
        entropy, the hamming distance list, and the length of retrieval results
    Output:
        bag_summary (list): The same format as input one but without low quality result
        (i.e, result with large hamming distance)
        top5_hamming_distance (float): The mean of average hamming distance in top 5
        retrival results of all mosaics
    """
    LOW_FREQ_THRSH = 3
    LOW_PRECENT_THRSH = 5
    HIGH_PERCENT_THRSH = 95
    len_info = [b[-1] for b in bag_summary]
    if len(set(len_info)) <= LOW_FREQ_THRSH:
        pass
    else:
        bag_summary = [b for b in bag_summary if b[-1]
                       > np.percentile(len_info, LOW_PRECENT_THRSH)
                       and b[-1] < np.percentile(len_info, HIGH_PERCENT_THRSH)]

    # Remove the mosaic if its top5 mean hammign distance is bigger than average
    top5_hamming_dist = np.mean([np.mean(b[2][0:5]) for b in bag_summary])

    bag_summary = sorted(bag_summary, key=lambda x: (x[1]))  # sort by certainty
    bag_summary = [b for b in bag_summary if np.mean(b[2][0:5]) <= top5_hamming_dist]
    return bag_summary, top5_hamming_dist


def Filtered_BY_Prediction(bag_summary, label_count_summary):
    """
    Implementation of Filtered_By_Prediction in the paper
    Input:
        bag_summary (list): The same as the output from Clean
        label_count_summary (dict): The dictionary storing the diagnosis occurrence 
        of the retrieval result in each mosaic
    Output:
        bag_removed: The index (positional) of moaic that should not be considered 
        among the top5
    """
    voting_board = defaultdict(float)
    for b in bag_summary[0:5]:
        bag_index = b[0]
        for k, v in label_count_summary[bag_index].items():
            voting_board[k] += v
    final_vote_candidates = sorted(voting_board.items(), key=lambda x: -x[1])
    fv_pointer = 0
    while True:
        final_vote = final_vote_candidates[fv_pointer][0]
        bag_removed = {}
        for b in bag_summary[0:5]:
            bag_index = b[0]
            max_vote = max(label_count_summary[bag_index].items(), key=operator.itemgetter(1))[0]
            if max_vote != final_vote:
                bag_removed[bag_index] = 1
        if len(bag_removed) != len(bag_summary[0:5]):
            break
        else:
            fv_pointer += 1
    return bag_removed