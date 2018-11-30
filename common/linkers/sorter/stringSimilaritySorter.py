import jellyfish
import numpy as np
from common.utils import *


class StringSimilaritySorter:
    @profile
    def sort(self, surface, question, candidates):
        candidates_similarity = np.array([jellyfish.levenshtein_distance(surface, candidate[1].lower())
                                          for candidate in candidates])
        max_distance = len(surface)
        filter_threshold = candidates_similarity < max_distance
        candidates_similarity = candidates_similarity[filter_threshold]
        filtered_candidates = np.array(candidates, dtype=object)[filter_threshold]
        idxs = np.argsort(candidates_similarity)

        return filtered_candidates[idxs]
