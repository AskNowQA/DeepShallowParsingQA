import numpy as np
from common.utils import *


class StringSimilaritySorter:
    def __init__(self, metric):
        self.metric = metric

    @profile
    def sort(self, surface, question, candidates):
        if candidates is None or len(candidates) == 0:
            return []
        candidates_distance = np.array(
            [(self.metric(surface, candidate[1].lower()) if len(candidate[1]) > 2 else 1)
             for candidate in candidates], dtype=float)
        exact_match_idx = [idx for idx, candidate in enumerate(candidates) if surface in candidate[1].lower()]
        candidates_distance[exact_match_idx] /= 2
        candidates_distance = candidates_distance
        filtered_candidates = np.array(candidates, dtype=object)
        idxs = np.argsort(candidates_distance)
        return filtered_candidates[idxs]
