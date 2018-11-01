from common.linkers.unorderedLinker import UnorderedLinker


class OrderedLinker(UnorderedLinker):
    def __init__(self, rel2id_path, core_chains_path, sorter, dataset):
        super(OrderedLinker, self).__init__(rel2id_path, core_chains_path, dataset)
        self.sorter = sorter

    def link(self, surface, question):
        unordered_results = super(OrderedLinker, self).link(surface, question)
        ordered_results = self.sorter.sort(surface, question, unordered_results)
        return ordered_results

    def best_ranks(self, surfaces, qarow):
        output = self.link_all(surfaces, qarow)
        if len(output) == 0:
            return -1
        output2 = []
        for relation in qarow.sparql.relations:
            for candidates_idx, candidates in enumerate(output):
                number_of_candidates = len(candidates)
                for idx, candidate in enumerate(candidates):
                    if candidate[0] == relation:
                        output2.append([relation, candidates_idx, 1 - idx / number_of_candidates])
        output2.sort(key=lambda x: x[2], reverse=True)
        used_relations = []
        used_candidates = []
        scores = []
        for item in output2:
            if item[0] in used_relations:
                continue
            elif item[1] in used_candidates:
                continue
            else:
                used_relations.append(item[0])
                used_candidates.append(item[1])
                scores.append(item[2])

        return sum(scores) / max(len(qarow.sparql.relations), len(surfaces))
