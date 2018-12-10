import numpy as np
from common.utils import *


class StringSimilaritySorter:
    def __init__(self, metric, return_similarity=False):
        self.metric = metric
        self.return_similarity = return_similarity

    @profile
    def sort(self, surface, question, candidates):
        if candidates is None or len(candidates) == 0:
            return []
        candidates_distance = np.array(
            [(self.metric(surface, candidate[1].lower()) if len(candidate[1]) > 2 else 1)
             for candidate in candidates], dtype=float)
        if self.return_similarity:
            max_distance = np.max(candidates_distance)
        exact_match_idx = [idx for idx, candidate in enumerate(candidates) if surface in candidate[1].lower()]
        candidates_distance[exact_match_idx] /= 2
        candidates_distance = candidates_distance
        filtered_candidates = np.array(candidates, dtype=object)
        idxs = np.argsort(candidates_distance)
        if self.return_similarity:
            candidates_similarity = 1 - (candidates_distance) / max(max_distance, len(surface))
            output = np.hstack((filtered_candidates[idxs], candidates_similarity[idxs].reshape(-1, 1)))
        else:
            output = filtered_candidates[idxs]
        return output
