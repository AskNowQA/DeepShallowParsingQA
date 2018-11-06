import jellyfish


class StringSimilaritySorter:
    def sort(self, surface, question, candidates):
        candidates_similarity = [[candidate, jellyfish.levenshtein_distance(surface, candidate[1].lower())]
                                 for candidate in candidates]
        candidates_similarity.sort(key=lambda x: x[1])

        return [candidate[0] for candidate in candidates_similarity]
