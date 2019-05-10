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
        self.history = {}

    @profile
    def link(self, surfaces, extra_surfaces, surface, question, extra_candidates):
        string_surface = ' '.join(surface)
        unordered_results = []
        if extra_candidates is not None and len(extra_candidates) > 0:  # False and
            unordered_results = extra_candidates
        else:
            unordered_results = self.candidate_generator.generate(surfaces, extra_surfaces, string_surface, question)
        if len(self.sorters) > 0:
            return [[surface, sorter.sort(string_surface, question, unordered_results)] for sorter in self.sorters]
        else:
            return [[surface, unordered_results]]

    def link_all(self, surfaces, extra_surfaces, question, extra_candidates):
        output = []
        for surface in surfaces:
            output.append(self.link(surfaces, extra_surfaces, surface, question, extra_candidates))
        return output

    @profile
    def best_ranks(self, surfaces, extra_surfaces, target_uris, question, k, train, extra_candidates=None):
        mrr = 0
        # if train:
        #     if (len(surfaces) != len(target_uris)) or any(
        #             [self.vocab.special[0] in item for item in surfaces]):
        #         return -1, mrr
        output = self.link_all(surfaces, extra_surfaces, question, extra_candidates)
        output = [item for tmp in output for item in tmp]
        if len(output) == 0:
            return [0] * len(surfaces), -1, mrr, []
        candidates_dict = [
            {candidate[0]: idx for idx, candidate in
             zip(range(len(candidates[1]) - 1, -1, -1), reversed(candidates[1]))}
            for candidates in output]
        output2 = []
        change_target_uris = []
        target_raw_uris = [target_uri.raw_uri for target_uri in target_uris]
        not_found_target_uri = list(target_raw_uris)
        # for k in range(2):
        for target_uri in not_found_target_uri:
            found = False
            for candidates_idx, candidates in enumerate(output):
                surface = candidates[0]
                candidates = candidates[1]
                number_of_candidates = len(candidates)
                if target_uri in candidates_dict[candidates_idx]:
                    idx = candidates_dict[candidates_idx][target_uri]
                    while idx >= 1 and output[candidates_idx][1][idx][1].lower() == output[candidates_idx][1][idx - 1][
                        1].lower():
                        idx -= 1
                    score = 1 - idx / number_of_candidates
                    if self.include_similarity_score and isinstance(output[candidates_idx][1][idx][-1], float):
                        score *= output[candidates_idx][1][idx][-1]
                    output2.append([target_uri, candidates_idx, score, idx, surface])
                    found = True
            # if not found and target_uri not in change_target_uris:
            #     if '/property/' in target_uri:
            #         new_uri = target_uri.replace('/property/', '/ontology/')
            #         not_found_target_uri.append(new_uri)
            #         change_target_uris.append(new_uri)
            #     elif '/ontology/' in target_uri:
            #         new_uri = target_uri.replace('/ontology/', '/property/')
            #         not_found_target_uri.append(new_uri)
            #         change_target_uris.append(new_uri)
        output2.sort(key=lambda x: x[2], reverse=True)

        if question not in self.history:
            self.history[question] = {}
        question_history = self.history[question]

        used_uris, used_candidates, used_surfaces, rank, found_uris = [], [], [], [], []
        scores = [[0] * len(surface) for surface in surfaces]
        for item in output2:
            surface_idx = surfaces.index(item[4])
            if item[0] in used_uris or item[1] in used_candidates or item[4] in used_surfaces:
                pass
            else:
                used_uris.append(item[0])
                used_candidates.append(item[1])
                used_surfaces.append(item[4])
                if item[0] in question_history:
                    best_score, best_surface = question_history[item[0]]
                    if max(best_score) > item[2]:
                        scores[surface_idx] = [max(best_score) if surface in best_surface else 0 for surface in
                                               item[4]]
                    else:
                        scores[surface_idx] = [item[2]] * len(item[4])
                        question_history[item[0]] = (scores[surface_idx], item[4])
                else:
                    scores[surface_idx] = [item[2]] * len(item[4])
                    question_history[item[0]] = (scores[surface_idx], item[4])

                if train:
                    if max(scores[surface_idx]) > 0.5 and item[3] <= k:
                        rank.append(item[3])
                elif item[3] <= k:
                    rank.append(item[3])
                    found_uris.append(item[0])
        max_len = max(len(target_uris), len(surfaces))
        if k >= 0 and max_len > 0:
            mrr = sum(map(lambda x: 1.0 / (x + 1), rank)) / max_len
        mean_score = sum(map(sum, scores)) / sum(map(len, scores))

        return scores, mean_score / max_len, mrr, found_uris

    @profile
    def ranked_link(self, surfaces, extra_surfaces, question, k, extra_candidates=None):
        output = self.link_all(surfaces, extra_surfaces, question, extra_candidates)
        output = [item for tmp in output for item in tmp]
        found_uris = [item[1][0][0] for item in output]

        output = [{'surface': [question.lower().index(item[0][0]), len(' '.join(item[0]))],
                   'uris': [{'confidence': uri[-1], 'uri': uri[0]} for uri in item[1][:k]]}
                  for item in
                  output]

        return output, found_uris
