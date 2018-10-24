from common.linkers.unorderedLinker import UnorderedLinker


class OrderedLinker(UnorderedLinker):
    def __init__(self, rel2id_path, core_chains_path, sorter, dataset):
        super(OrderedLinker, self).__init__(rel2id_path, core_chains_path, dataset)
        self.sorter = sorter

    def link(self, surface, question):
        unordered_results = super(OrderedLinker, self).link(surface, question)
        ordered_results = self.sorter.sort(surface, question, unordered_results)
        return ordered_results
