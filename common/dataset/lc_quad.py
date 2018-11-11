from common.dataset.container.qarow import QARow
from common.vocab import Vocab
from config import config

import ujson as json
import os
import pickle as pk
import torch


class LC_QuAD:
    def __init__(self, trianset_path, testset_path, vocab_path, remove_entity_mention=False, remove_stop_words=False):
        self.train_set, self.train_corpus = self.__load_dataset(trianset_path, remove_entity_mention, remove_stop_words)
        self.test_set, self.test_corpus = self.__load_dataset(testset_path, remove_entity_mention, remove_stop_words)

        self.corpus = self.train_corpus + self.test_corpus
        if not os.path.isfile(vocab_path):
            self.__build_vocab(self.corpus, vocab_path)
        self.__load_candidate_relations(vocab_path)
        self.vocab = Vocab(filename=vocab_path, data=['<ent>'])

        self.coded_train_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.train_corpus]
        self.coded_test_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.test_corpus]

    def __load_dataset(self, dataset_path, remove_entity_mention, remove_stop_words):
        if not os.path.isfile(dataset_path):
            return [], []
        with open(dataset_path, 'r') as file_hanlder:
            raw_dataset = json.load(file_hanlder)

            dataset = [QARow(item['corrected_question'],
                             item['annotation'] if 'annotation' in item else '',
                             item['sparql_query'],
                             remove_entity_mention, remove_stop_words)
                       for item in
                       raw_dataset]
            # if len(re.findall('<[^>]*>', item['sparql_query'])) <= 2]
            dataset = [row for row in dataset if len(row.sparql.relations) == 1 and len(row.sparql.entities) == 1]
            corpus = [item.normalized_question for item in dataset]
            return dataset, corpus

    def __load_candidate_relations(self, vocab_path):
        with open(config['lc_quad']['rel2id'], 'rb') as f_h:
            rel2id = pk.load(f_h, encoding='latin1')

        if os.path.isfile(config['lc_quad']['rel_vocab']):
            vocab = Vocab(filename=vocab_path, data=['<ent>'])
            vocab.loadFile(config['lc_quad']['rel_vocab'])
        else:
            vocab = set()
            for item_id, item in rel2id.items():
                words = [word.lower().replace('.', '') for word in item[2]]
                vocab |= set(words)
            print(len(vocab))
            with open(config['lc_quad']['rel_vocab'], 'w') as f:
                for token in sorted(vocab):
                    f.write(token + '\n')
            vocab = Vocab(filename=vocab_path, data=['<ent>'])
            vocab.loadFile(config['lc_quad']['rel_vocab'])

        ## Need to fix cases where there are non-alphabet chars in the label
        for item_id, item in rel2id.items():
            idxs = [vocab.getIndex(word.lower().replace('.', '')) for word in item[2]]
            idxs = [id for id in idxs if id is not None]
            idxs = torch.LongTensor(idxs)
            if len(item) >= 6:
                item[5] = idxs
                if len(item) >= 7:
                    del item[6:]
            else:
                item.append(idxs)

        with open(config['lc_quad']['rel2id'], 'wb') as f_h:
            pk.dump(rel2id, f_h)

    def __build_vocab(self, lines, vocab_path):
        vocab = set()
        for tokens in lines:
            vocab |= set(tokens)
        if '<ent>' in vocab:
            vocab.remove('<ent>')
        with open(vocab_path, 'w') as f:
            for token in sorted(vocab):
                f.write(token + '\n')
