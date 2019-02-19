import numpy as np
from common.utils import *


class StringSimilaritySorter:
    def __init__(self, metric, metric_range_percentage=False, return_similarity=False):
        self.metric = metric
        self.metric_range_percentage = metric_range_percentage
        self.return_similarity = return_similarity

    @profile
    def sort(self, surface, question, candidates):
        if candidates is None or len(candidates) == 0:
            return []
        candidates_distance = np.array(
            [(self.metric(surface, candidate[1].lower()))
             for candidate in candidates], dtype=float)
        exact_match_idx = [idx for idx, candidate in enumerate(candidates) if surface in candidate[1].lower()]
        candidates_distance[exact_match_idx] /= 2
        candidates_distance = candidates_distance
        filtered_candidates = np.array(candidates, dtype=object)
        idxs = np.argsort(candidates_distance)
        if self.return_similarity:
            if self.metric_range_percentage:
                candidates_similarity = 1 - candidates_distance
            else:
                surface_len = len(surface)
                candidates_len = np.array([max(surface_len, len(candidate[1])) for candidate in candidates])
                candidates_similarity = 1 - (candidates_distance / candidates_len)
            output = np.hstack((filtered_candidates[idxs], candidates_similarity[idxs].reshape(-1, 1)))
        else:
            output = filtered_candidates[idxs]
        return output
