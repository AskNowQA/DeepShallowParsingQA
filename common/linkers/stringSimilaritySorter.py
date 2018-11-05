import jellyfish


class StringSimilaritySorter:
    def sort(self, surface, question, relations):
        relations_similarity = [[relation, jellyfish.levenshtein_distance(surface.lower(), relation[1].lower())]
                                for relation in relations]
        relations_similarity.sort(key=lambda x: x[1])

        return [relation[0] for relation in relations_similarity]
