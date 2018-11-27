import jellyfish
from common.utils import *

class StringSimilaritySorter:
    @profile
    def sort(self, surface, question, candidates):
        candidates_similarity = [[candidate, jellyfish.levenshtein_distance(surface, candidate[1].lower())]
                                 for candidate in candidates]
        max_distance = len(surface)
        candidates_similarity = [item for item in candidates_similarity if item[1] < max_distance]
        candidates_similarity.sort(key=lambda x: x[1])

        return [candidate[0] for candidate in candidates_similarity]
