from common.linkers.orderedLinker import OrderedLinker
from common.utils import *


class EntityOrderedLinker(OrderedLinker):
    def __init__(self, candidate_generator, sorters, vocab, include_similarity_score=True):
        super(EntityOrderedLinker, self).__init__(candidate_generator, sorters, vocab, include_similarity_score)

    @profile
    def best_ranks(self, surfaces, extra_surfaces, qa_row, k, train):
        # if train:
        #     for idx, surface in enumerate(surfaces):
        #         keep = False
        #         for entity in qa_row.sparql.entities:
        #             string_surface = ' '.join(self.vocab.convertToLabels(surface))
        #             surface_ngram = Utils.ngrams(string_surface)
        #             if len(surface_ngram) > 0 and len(entity.ngram.intersection(surface_ngram)) / len(
        #                     surface_ngram) >= 0.4:
        #                 keep = True
        #                 break
        #         if not keep:
        #             surfaces[idx] = []

        results = super(EntityOrderedLinker, self).best_ranks(surfaces,
                                                              extra_surfaces,
                                                              qa_row.sparql.entities,
                                                              qa_row.question,
                                                              k,
                                                              train)

        self.logger.debug([' '.join(item) for item in surfaces])
        self.logger.debug([rel.raw_uri for rel in qa_row.sparql.entities])
        self.logger.debug(list(map('{:0.2f}'.format, results[1:-1])))

        return results
