from common.linkers.orderedLinker import OrderedLinker


class EntityOrderedLinker(OrderedLinker):
    def __init__(self, candidate_generator, sorters, vocab):
        super(EntityOrderedLinker, self).__init__(candidate_generator, sorters, vocab)

    def best_ranks(self, surfaces, qa_row, k, train):
        results = super(EntityOrderedLinker, self).best_ranks(surfaces, qa_row.sparql.entities, qa_row.question, k, train)

        self.logger.debug(qa_row.question)
        self.logger.debug(qa_row.normalized_question)
        self.logger.debug([self.vocab.convertToLabels(item) for item in surfaces])
        self.logger.debug([rel.raw_uri for rel in qa_row.sparql.relations])

        return results
