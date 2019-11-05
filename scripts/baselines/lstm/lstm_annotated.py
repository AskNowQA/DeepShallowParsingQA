import os
import ujson as json
import numpy as np
import torch

from scripts.baselines.lstm.lstm import *
from config import config
from common.dataset.lc_quad import LC_QuAD
from common.linkers.candidate_generator.datasetCG import DatasetCG
from common.linkers.sorter.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.sorter.embeddingSimilaritySorter import EmbeddingSimilaritySorter
from common.model.environment import Environment
from common.linkers.entityOrderedLinker import EntityOrderedLinker
from common.linkers.relationOrderLinker import RelationOrderedLinker
from common.linkers.candidate_generator.elasticCG import ElasticCG
from common.linkers.candidate_generator.elastic import Elastic

if __name__ == '__main__':
    with open(os.path.join(config['lc_quad']['base_path'], 'lcquad_annotated.json'), 'rt') as f:
        dataset_annotated_ = json.load(f)

    dataset_annotated = {item['question']: item for item in dataset_annotated_}
    dataset_annotated2 = {item['sparql_query'].replace('\u003c', '<').replace('\u003e', '>'): item for item in
                          dataset_annotated_}

    k = 10
    ds_name = 'lcquad-annotated'
    dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      False, False)
    word_vectorizer = dataset.word_vectorizer

    train_set_file_name = 'prep-{}.json'.format(ds_name)
    if os.path.exists(train_set_file_name):
        train_set = torch.load(train_set_file_name)
    else:
        train_set = {}
        for idx, item in enumerate(dataset.coded_train_corpus):

            qarow = dataset.train_set[idx]

            if qarow.question in dataset_annotated:
                annotations = dataset_annotated[qarow.question]
            elif qarow.sparql.raw_sparql in dataset_annotated2:
                annotations = dataset_annotated2[qarow.sparql.raw_sparql]
            else:
                print(qarow.question)
                continue

            masked = '0' * len(annotations['question'])
            for ent in annotations['entity mapping']:
                start, end = (map(int, ent['seq'].split(',')))
                masked = masked[:start] + '2' * (end - start) + masked[end:]
            for ent in annotations['predicate mapping']:
                start, end = (map(int, ent['seq'].split(',')))
                masked = masked[:start] + '1' * (end - start) + masked[end:]

            labels = []
            words = []
            tmp = []
            for idx, word in enumerate(qarow.normalized_question):
                word_coded = dataset.vocab.getIndex(word.lower())
                words.append(torch.LongTensor([word_coded]))
                label = np.rint(np.mean(list(map(int, masked[len(tmp):len(tmp) + len(word)]))))
                if np.isnan(label):
                    label = 0
                labels.append(label)
                tmp += word + ' '
            train_set[qarow.question] = [torch.LongTensor(words), torch.LongTensor(labels)]
        torch.save(train_set, train_set_file_name)

    classifier = LSTMClassifier(vocab_size=dataset.vocab.size(),
                                emb_size=dataset.word_vectorizer.word_size,
                                input_size=dataset.word_vectorizer.word_size,
                                hidden_size=dataset.word_vectorizer.word_size,
                                output_size=3,
                                dropout_ratio=0.1,
                                emb_idx=0,
                                bidirectional=False)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.1)

    checkpoint_filename = ds_name + '.chk'
    if os.path.isfile(checkpoint_filename):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_filename)
        else:
            checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(checkpoint['model'])
    else:
        train(train_set.values(), classifier, loss_function, optimizer, 100)
        checkpoint = {'model': classifier.state_dict()}
        torch.save(checkpoint, checkpoint_filename)

    elastic = Elastic(config['elastic']['server'])
    entity_linker = EntityOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='entity_whole_match_index'),
        sorters=[StringSimilaritySorter(similarity.ngram.NGram(2).distance, True)],
        vocab=dataset.vocab)
    relation_linker = RelationOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='relation_whole_match_index'),
        sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, False, True),
                 EmbeddingSimilaritySorter(dataset.word_vectorizer)],
        vocab=dataset.vocab)
    eval(dataset, train_set, classifier, entity_linker, relation_linker, loss_function)

## lstm train§
# 0.5195291666666667
# 0.3212166666666667