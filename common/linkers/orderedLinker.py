import logging
import numpy as np
from common.utils import *


class OrderedLinker:
    def __init__(self, candidate_generator, sorters, vocab, include_similarity_score=False):
        self.candidate_generator = candidate_generator
        self.sorters = sorters
        self.vocab = vocab
        self.logger = logging.getLogger('main')
        self.include_similarity_score = include_similarity_score

    @profile
    def link(self, surface, question):
        string_surface = ' '.join(self.vocab.convertToLabels(surface))
        unordered_results = self.candidate_generator.generate(string_surface, question)
        return [[surface, sorter.sort(string_surface, question, unordered_results)] for sorter in self.sorters]

    def link_all(self, surfaces, question):
        output = []
        for surface in surfaces:
            output.append(self.link(surface, question))
        return output

    @profile
    def best_ranks(self, surfaces, target_uris, question, k, train):
        mrr = 0
        if train:
            if (len(surfaces) != len(target_uris)) or any(
                    [self.vocab.special[0] in item for item in surfaces]):
                return -1, mrr
        output = self.link_all(surfaces, question)
        output = [item for tmp in output for item in tmp]
        if len(output) == 0:
            return -1, mrr
        candidates_dict = [
            {candidate[0]: idx for idx, candidate in
             zip(range(len(candidates[1]) - 1, -1, -1), reversed(candidates[1]))}
            for candidates in output]
        output2 = []
        for target_uri in target_uris:
            for candidates_idx, candidates in enumerate(output):
                surface = candidates[0]
                candidates = candidates[1]
                number_of_candidates = len(candidates)
                if target_uri.raw_uri in candidates_dict[candidates_idx]:
                    idx = candidates_dict[candidates_idx][target_uri.raw_uri]
                    if idx >= 1 and output[candidates_idx][1][idx][1] == output[candidates_idx][1][idx - 1][1]:
                        idx -= 1
                    output2.append([target_uri, candidates_idx, 1 - idx / number_of_candidates, idx, surface])
        output2.sort(key=lambda x: x[2], reverse=True)
        used_uris, used_candidates, scores, rank = [], [], [], []
        for item in output2:
            if item[0] in used_uris or item[1] in used_candidates:
                continue
            else:
                used_uris.append(item[0])
                used_candidates.append(item[1])
                tmp = 1
                # tmp = len(item[0].tokens) / (abs(len(item[4]) - len(item[0].tokens)) + 1)
                if self.include_similarity_score and isinstance(output[item[1]][1][item[3]][-1], float):
                    tmp = output[item[1]][1][item[3]][-1]
                scores.append(item[2] * tmp)
                if item[3] <= k:
                    rank.append(item[3])
        max_len = max(len(target_uris), len(surfaces))
        if k >= 0 and max_len > 0:
            mrr = sum(map(lambda x: 1.0 / (x + 1), rank)) / max_len

        return sum(scores) / max_len, mrr
