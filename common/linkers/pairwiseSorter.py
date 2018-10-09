class PairwiseSorter:
    def __init__(self, pairwise_ranker):
        self.ranker = pairwise_ranker

    def sort(self, surface, question, relations):
        unordered_results = relations
        exchanges = True
        passnum = len(unordered_results) - 1
        while passnum > 0 and exchanges:
            exchanges = False
            for i in range(passnum):
                if self.ranker.rank(surface, question, unordered_results[i], unordered_results[i + 1]):
                    exchanges = True
                    temp = unordered_results[i]
                    unordered_results[i] = unordered_results[i + 1]
                    unordered_results[i + 1] = temp
            passnum = passnum - 1

        return unordered_results