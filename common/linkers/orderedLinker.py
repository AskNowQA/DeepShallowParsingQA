from common.linkers.unorderedLinker import UnorderedLinker


class OrderedLinker(UnorderedLinker):
    def __init__(self, rel2id_path, core_chains_path, sorter, dataset):
        super(OrderedLinker, self).__init__(rel2id_path, core_chains_path, dataset)
        self.sorter = sorter

    def link(self, surface, question):
        unordered_results = super(OrderedLinker, self).link(surface, question)
        ordered_results = self.sorter.sort(surface, question, unordered_results)
        return ordered_results

    def best_ranks(self, surfaces, qarow, k):
        output = self.link_all(surfaces, qarow)
        mrr = 0
        if len(output) == 0:
            return -1, mrr
        output2 = []
        for relation in qarow.sparql.relations:
            for candidates_idx, candidates in enumerate(output):
                number_of_candidates = len(candidates)
                for idx, candidate in enumerate(candidates):
                    if candidate[0] == relation.raw_uri:
                        output2.append([relation, candidates_idx, 1 - idx / number_of_candidates, idx])
        output2.sort(key=lambda x: x[2], reverse=True)
        used_relations, used_candidates, scores, rank = [], [], [], []
        for item in output2:
            if item[0] in used_relations or item[1] in used_candidates:
                continue
            else:
                used_relations.append(item[0])
                used_candidates.append(item[1])
                scores.append(item[2])
                if item[3] <= k:
                    rank.append(item[3])
        max_len = max(len(qarow.sparql.relations), len(surfaces))
        if k > 0 and max_len > 0:
            mrr = sum(map(lambda x: 1.0 / (x + 1), rank)) / max_len
        return sum(scores) / max_len, mrr
