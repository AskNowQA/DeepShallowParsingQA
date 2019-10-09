import numpy as np
import jellyfish
import similarity.ngram
from common.linkers.entityOrderedLinker import EntityOrderedLinker
from common.linkers.sorter.embeddingSimilaritySorter import EmbeddingSimilaritySorter
from common.linkers.sorter.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.candidate_generator.elasticCG import ElasticCG
from common.linkers.candidate_generator.elastic import Elastic
from common.linkers.relationOrderLinker import RelationOrderedLinker
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.lc_quad import LC_QuAD
from config import config
from flair.data import Sentence
from flair.models import SequenceTagger

dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                  False, False)
# dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
#                     False, False)

chunk_tagger = SequenceTagger.load('chunk')
ner_tagger = SequenceTagger.load('ner')


def get_phrases(sentence):
    sentence = Sentence(sentence)
    # ner_tagger.predict(sentence)
    # entities = []
    # for item in sentence.get_spans('ner'):
    #     entities.append(item.text.split())
    # relations = []
    # return [relations, entities]

    chunk_tagger.predict(sentence)

    entities = []
    relations = []
    for item in sentence.get_spans('np'):
        if item.tag == 'NP':
            entities.append(item.text.split())
        elif item.tag == 'VP':
            relations.append(item.text.split())
    return [relations, entities]


if __name__ == '__main__':
    elastic = Elastic(config['elastic']['server'])
    entity_linker = EntityOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='entity_whole_match_index'),
        sorters=[StringSimilaritySorter(similarity.ngram.NGram(2).distance, True)],
        vocab=dataset.vocab)

    relation_linker = RelationOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='relation_whole_match_index'),
        sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, False, True),
                 # EmbeddingSimilaritySorter(dataset.word_vectorizer)
                 ],
        vocab=dataset.vocab)

    total_relation_rmm, total_entity_rmm = [], []
    for qarow in dataset.test_set:
        surfaces = get_phrases(qarow.question)

        entity_results, entity_score, entity_mrr, found_target_entities = entity_linker.best_ranks(
            list(surfaces[1]), list(surfaces[0]), qarow, 1, False)

        extra_candidates = []
        extra_candidates.extend(dataset.find_one_hop_relations(found_target_entities))

        relation_results, relation_score, relation_mrr, _ = relation_linker.best_ranks(
            list(surfaces[0]), list(surfaces[1]), qarow, 1, False, extra_candidates)
        total_entity_rmm.append(entity_mrr)
        total_relation_rmm.append(relation_mrr)

    print([np.mean(total_entity_rmm), np.mean(total_relation_rmm)])

# Q7
# FLAIR chunker
# [0.21066666666666667, 0.0]
# FLAIR ner
# [0.43]

#LC-QuAD
# FLAIR chunker
# [0.28, 0.05]
# FLAIR ner
# [0.59]
