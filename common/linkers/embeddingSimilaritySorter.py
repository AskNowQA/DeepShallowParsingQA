from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingSimilaritySorter:
    def __init__(self, word_vectorizer):
        self.word_vectorizer = word_vectorizer

    def sort(self, surface, question, candidates):
        surface_embeddings = self.word_vectorizer.decode(surface)
        candidates_similarity = [[candidate,
                                  np.mean(cosine_similarity(surface_embeddings,
                                                            self.word_vectorizer.vectors[candidate[5]]))]
                                 for candidate in candidates
                                 if len(candidate[5]) > 0]
        candidates_similarity.sort(key=lambda x: x[1], reverse=True)

        return [candidate[0] for candidate in candidates_similarity]
