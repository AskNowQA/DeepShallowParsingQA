import jellyfish
import numpy as np
from common.utils import *


class StringSimilaritySorter:
    @profile
    def sort(self, surface, question, candidates):
        candidates_similarity = np.array([jellyfish.levenshtein_distance(surface, candidate[1].lower())
                                          for candidate in candidates])
        max_distance = len(surface)
        candidates_similarity = candidates_similarity[candidates_similarity < max_distance]
        idxs = np.argsort(candidates_similarity)

        return [candidates[idx] for idx in idxs]
