import numpy as np
import torch
import torch.nn as nn
from common.utils import *


class EmbeddingSimilaritySorter:
    def __init__(self, word_vectorizer):
        self.word_vectorizer = word_vectorizer
        emb_shape = self.word_vectorizer.emb.shape
        self.emb = nn.Embedding(emb_shape[0], emb_shape[1], padding_idx=0, sparse=False)
        self.emb.weight.data.copy_(word_vectorizer.emb)
        if torch.cuda.is_available():
            self.emb.cuda()

    @profile
    def sort(self, surface, question, candidates):
        if len(candidates) == 0:
            return []
        surface_embeddings = self.word_vectorizer.decode(surface)
        surface_embeddings = torch.mean(surface_embeddings, dim=0).reshape(1, -1)

        tmp = [item[5] for item in candidates]
        lengths = [item[6] for item in candidates]
        lens = torch.FloatTensor(lengths).reshape(-1, 1)
        candidates_coded = torch.stack(tmp)

        if torch.cuda.is_available():
            surface_embeddings = surface_embeddings.cuda()
            candidates_coded = candidates_coded.cuda()
            lens = lens.cuda()
        candidates_embeddings = self.emb(candidates_coded)
        candidates_embeddings_mean = torch.sum(candidates_embeddings, dim=1) / lens
        candidates_similarity = torch.nn.functional.cosine_similarity(surface_embeddings, candidates_embeddings_mean)
        candidates_similarity = candidates_similarity.data
        if torch.cuda.is_available():
            candidates_similarity = candidates_similarity.cpu()
        candidates_similarity = candidates_similarity.numpy()
        sorted_idx = np.argsort(candidates_similarity)[::-1]
        sorted_idx = [idx for idx in sorted_idx if candidates_similarity[idx] > 0.4]
        sorted = [candidates[idx] for idx in sorted_idx]
        return sorted
