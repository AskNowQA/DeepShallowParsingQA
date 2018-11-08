from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingSimilaritySorter:
    def __init__(self, word_vectorizer):
        self.word_vectorizer = word_vectorizer

    def sort(self, surface, question, candidates):
        surface_embeddings = self.word_vectorizer.decode(surface)
        candidates_embeddings = [[idx, surface_embeddings, self.word_vectorizer.vectors[candidate[5]]]
                                 for idx, candidate in enumerate(candidates)
                                 if len(candidate[5]) > 0]
        candidates_similarity = [[item[0], np.mean(cosine_similarity(item[1], item[2]))] for item in
                                 candidates_embeddings]
        candidates_similarity.sort(key=lambda x: x[1], reverse=True)

        return [candidates[item[0]] for item in candidates_similarity]
