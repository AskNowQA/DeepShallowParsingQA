from common.linkers.orderedLinker import OrderedLinker
from common.utils import *


class EntityOrderedLinker(OrderedLinker):
    def __init__(self, candidate_generator, sorters, vocab):
        super(EntityOrderedLinker, self).__init__(candidate_generator, sorters, vocab)

    @profile
    def best_ranks(self, surfaces, qa_row, k, train):
        if train:
            for entity in qa_row.sparql.entities:
                for surface in list(surfaces):
                    string_surface = ' '.join(self.vocab.convertToLabels(surface))
                    surface_ngram = Utils.ngrams(string_surface)
                    if len(surface_ngram) == 0 or len(entity.ngram.intersection(surface_ngram)) / len(
                            surface_ngram) < 0.7:
                        surfaces.remove(surface)

        results = super(EntityOrderedLinker, self).best_ranks(surfaces,
                                                              qa_row.sparql.entities,
                                                              qa_row.question,
                                                              k,
                                                              train)

        # self.logger.debug(qa_row.question)
        # self.logger.debug(qa_row.normalized_question)
        self.logger.debug([' '.join(self.vocab.convertToLabels(item)) for item in surfaces])
        self.logger.debug([rel.raw_uri for rel in qa_row.sparql.entities])
        self.logger.debug(results[1])

        return results
