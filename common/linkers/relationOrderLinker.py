from common.linkers.orderedLinker import OrderedLinker


class RelationOrderedLinker(OrderedLinker):
    def __init__(self, candidate_generator, sorters, vocab, include_similarity_score=True):
        super(RelationOrderedLinker, self).__init__(candidate_generator, sorters, vocab, include_similarity_score)

    def best_ranks(self, surfaces, qa_row, k, train, extra_candidates=None):
        results = super(RelationOrderedLinker, self).best_ranks(surfaces,
                                                                qa_row.sparql.relations,
                                                                qa_row.question,
                                                                k,
                                                                train,
                                                                extra_candidates)

        self.logger.debug([' '.join(item) for item in surfaces])
        self.logger.debug([rel.raw_uri for rel in qa_row.sparql.relations])
        self.logger.debug(list(map('{:0.2f}'.format, results[1:-1])))

        return results
