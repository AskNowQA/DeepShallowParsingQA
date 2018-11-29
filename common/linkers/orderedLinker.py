import logging
from common.linkers.unorderedLinker import UnorderedLinker
from common.utils import *


class OrderedLinker(UnorderedLinker):
    def __init__(self, rel2id_path, core_chains_path, sorters, dataset):
        super(OrderedLinker, self).__init__(rel2id_path, core_chains_path, dataset)
        self.sorters = sorters
        self.logger = logging.getLogger('main')

    @profile
    def link(self, surface, question):
        string_surface = ' '.join(self.dataset.vocab.convertToLabels(surface))
        unordered_results = super(OrderedLinker, self).link(string_surface, question)
        return [[surface, sorter.sort(string_surface, question, unordered_results)] for sorter in self.sorters]

    @profile
    def best_ranks(self, surfaces, qarow, k, train):
        mrr = 0
        if train:
            if (len(surfaces) != len(qarow.sparql.relations)) or any(
                    [self.dataset.vocab.special[0] in item for item in surfaces]):
                return -1, mrr
        output = self.link_all(surfaces, qarow)
        output = [item for tmp in output for item in tmp]
        if len(output) == 0:
            return -1, mrr
        candidates_dict = [{candidate[0]: idx for idx, candidate in enumerate(candidates[1])} for candidates in output]
        output2 = []
        for relation in qarow.sparql.relations:
            for candidates_idx, candidates in enumerate(output):
                surface = candidates[0]
                candidates = candidates[1]
                number_of_candidates = len(candidates)
                if relation.raw_uri in candidates_dict[candidates_idx]:
                    idx = candidates_dict[candidates_idx][relation.raw_uri]
                    output2.append([relation, candidates_idx, 1 - idx / number_of_candidates, idx, surface])
        output2.sort(key=lambda x: x[2], reverse=True)
        used_relations, used_candidates, scores, rank = [], [], [], []
        for item in output2:
            if item[0] in used_relations or item[1] in used_candidates:
                continue
            else:
                used_relations.append(item[0])
                used_candidates.append(item[1])
                tmp = len(item[0].tokens) / (abs(len(item[4]) - len(item[0].tokens)) + 1)
                scores.append(item[2] * tmp)
                if item[3] <= k:
                    rank.append(item[3])
        max_len = max(len(qarow.sparql.relations), len(surfaces))
        if k >= 0 and max_len > 0:
            mrr = sum(map(lambda x: 1.0 / (x + 1), rank)) / max_len
        self.logger.debug(qarow.question)
        self.logger.debug(qarow.normalized_question)
        self.logger.debug([' '.join(self.dataset.vocab.convertToLabels(item)) for item in surfaces])
        self.logger.debug([rel.raw_uri for rel in qarow.sparql.relations])

        return sum(scores) / max_len, mrr
