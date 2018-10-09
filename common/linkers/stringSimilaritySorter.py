from difflib import SequenceMatcher


class StringSimilaritySorter:
    def sort(self, surface, question, relations):
        relations_similarity = [[relation, SequenceMatcher(None, surface, relation[1]).ratio()]
                                for relation in relations]
        relations_similarity.sort(key=lambda x: x[1], reverse=True)

        return [relation[0] for relation in relations_similarity]
