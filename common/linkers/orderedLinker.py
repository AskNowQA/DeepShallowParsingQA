from common.linkers.unorderedLinker import UnorderedLinker


class OrderedLinker(UnorderedLinker):
    def __init__(self, dataset_path, relations_path, sorter):
        super(OrderedLinker, self).__init__(dataset_path, relations_path)
        self.sorter = sorter

    def link(self, surface, question):
        unordered_results = super(OrderedLinker, self).link(surface, question)
        ordered_results = self.sorter.sort(surface, question, unordered_results)

        return ordered_results
