from common.word_vectorizer.glove import Glove
from common.vocab import Vocab
from config import config

import torch
import torch.nn as nn
import os
import pickle as pk


class Base_Dataset:
    def __init__(self, trainset_path, testset_path, vocab_path, dataset_name='', remove_entity_mention=False,
                 remove_stop_words=False):
        self.config = config[dataset_name]
        self.train_set, self.train_corpus = self.load_dataset(trainset_path, remove_entity_mention, remove_stop_words)
        self.test_set, self.test_corpus = self.load_dataset(testset_path, remove_entity_mention, remove_stop_words)

        self.corpus = self.train_corpus + self.test_corpus
        if not os.path.isfile(vocab_path):
            self.__build_vocab(self.corpus, vocab_path)
        self.vocab = Vocab(filename=vocab_path, data=['<ent>', '<num>'])
        self.word_vectorizer = Glove(self.vocab, config['glove_path'], self.config['emb'])

        for qa_row in self.train_set + self.test_set:
            for relation in qa_row.sparql.relations:
                relation.coded = self.decode(relation)
        # self.__update_relations_emb()

        self.coded_train_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.train_corpus]
        self.coded_test_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.test_corpus]
        self.vocab_path = vocab_path

        self.one_hop = None
        if os.path.isfile(self.config['entity_one_hop']):
            with open(self.config['entity_one_hop'], 'rb') as f:
                self.one_hop = pk.load(f)

    def decode(self, relation, max_length=3):
        idxs = self.vocab.convertToIdx(map(str.lower, relation.tokens[:max_length]), '')
        length = len(idxs)
        if len(idxs) < max_length:
            idxs = idxs + [0] * (max_length - len(idxs))
        return torch.LongTensor(idxs), length

    def load_dataset(self, dataset_path, remove_entity_mention, remove_stop_words):
        return [], []

    def __load_candidate_relations(self):
        with open(self.config['rel2id'], 'rb') as f_h:
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

        with open(self.config['rel2id'], 'rb') as f_h:
            rel2id = pk.load(f_h, encoding='latin1')

        ## Need to fix cases where there are non-alphabet chars in the label
        max_length = 3
        for item_id, item in rel2id.items():
            if len(item[2]) > max_length:
                idxs = []
            else:
                idxs = [self.vocab.getIndex(
                    word.lower().replace('.', '') if not word.replace('.', '').replace('(', '').isdigit() else '<num>')
                    for word in item[2]]
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
        with open(self.config['rel2id'], 'wb') as f_h:
            pk.dump(rel2id, f_h)

    def __build_vocab(self, lines, vocab_path):
        vocab = set()
        for tokens in lines:
            vocab |= set(tokens)
        relations_vocab = self.__load_candidate_relations()
        vocab |= relations_vocab
        vocab = [w for w in vocab if not w.replace('.', '').replace('(', '').isdigit()]
        if '<ent>' in vocab:
            vocab.remove('<ent>')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token in sorted(vocab):
                f.write(token + '\n')

    def find_one_hop_relations(self, entities):
        extra_candidates = []
        if self.one_hop is not None:
            for entity in entities:
                if entity in self.one_hop:
                    extra_candidates.extend(self.one_hop[entity])
        return extra_candidates
