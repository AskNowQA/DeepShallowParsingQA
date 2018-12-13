from common.dataset.container.qarow import QARow
from common.word_vectorizer.glove import Glove
from common.vocab import Vocab
from config import config

import torch
import torch.nn as nn
import ujson as json
import os
import pickle as pk


class LC_QuAD:
    def __init__(self, trianset_path, testset_path, vocab_path, remove_entity_mention=False, remove_stop_words=False):
        self.train_set, self.train_corpus = self.__load_dataset(trianset_path, remove_entity_mention, remove_stop_words)
        self.test_set, self.test_corpus = self.__load_dataset(testset_path, remove_entity_mention, remove_stop_words)

        self.corpus = self.train_corpus + self.test_corpus
        if not os.path.isfile(vocab_path):
            self.__build_vocab(self.corpus, vocab_path)
        self.vocab = Vocab(filename=vocab_path, data=['<ent>'])
        self.word_vectorizer = Glove(self.vocab, config['glove_path'], config['lc_quad']['emb'])
        # self.__update_relations_emb()

        self.coded_train_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.train_corpus]
        self.coded_test_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.test_corpus]
        self.vocab_path = vocab_path

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
            # with open('/Users/hamid/workspace/DeepShallowParsingQA/data/lcquad/no_constraints.json', 'r') as f:
            #     no_contraints = json.load(f)
            #     dataset = [row for row in dataset if row.question in no_contraints]
            dataset = [row for row in dataset if len(row.sparql.relations) == 1 and len(row.sparql.entities) == 1]
            corpus = [item.normalized_question for item in dataset]
            return dataset, corpus

    def __load_candidate_relations(self):
        with open(config['lc_quad']['rel2id'], 'rb') as f_h:
            rel2id = pk.load(f_h, encoding='latin1')

        vocab = set()
        for item_id, item in rel2id.items():
            words = [word.lower().replace('.', '') for word in item[2]]
            vocab |= set(words)

        return vocab

    def __update_relations_emb(self):
        emb_shape = self.word_vectorizer.emb.shape
        emb = nn.Embedding(emb_shape[0], emb_shape[1], padding_idx=0, sparse=False)
        emb.weight.data.copy_(self.word_vectorizer.emb)
        if torch.cuda.is_available():
            emb.cuda()

        with open(config['lc_quad']['rel2id'], 'rb') as f_h:
            rel2id = pk.load(f_h, encoding='latin1')

        ## Need to fix cases where there are non-alphabet chars in the label
        max_length = 3
        for item_id, item in rel2id.items():
            if len(item[2]) > max_length:
                idxs = []
            else:
                idxs = [self.vocab.getIndex(word.lower().replace('.', '')) for word in item[2]]
                idxs = [id for id in idxs if id is not None]
            length = len(idxs)
            if length == 0:
                length = 1
            if len(idxs) < max_length:
                idxs = idxs + [0] * (max_length - len(idxs))
            idxs = torch.LongTensor(idxs)
            item[5] = idxs
            if len(item) == 6:
                item.append(length)
            else:
                item[6] = length
        with open(config['lc_quad']['rel2id'], 'wb') as f_h:
            pk.dump(rel2id, f_h)

    def __build_vocab(self, lines, vocab_path):
        vocab = set()
        for tokens in lines:
            vocab |= set(tokens)
        relations_vocab = self.__load_candidate_relations()
        vocab |= relations_vocab
        if '<ent>' in vocab:
            vocab.remove('<ent>')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token in sorted(vocab):
                f.write(token + '\n')
