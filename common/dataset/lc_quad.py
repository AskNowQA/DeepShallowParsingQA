from common.dataset.container.qarow import QARow
from common.vocab import Vocab

import ujson as json
import os
import re


class LC_QuAD:
    def __init__(self, trianset_path, testset_path, vocab_path):
        self.train_set, self.train_corpus = self.__load_dataset(trianset_path)
        self.test_set, self.test_corpus = self.__load_dataset(testset_path)

        self.corpus = self.train_corpus + self.test_corpus
        if not os.path.isfile(vocab_path):
            self.__build_vocab(self.corpus, vocab_path)
        self.vocab = Vocab(filename=vocab_path)

        self.coded_train_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.train_corpus]
        self.coded_test_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.test_corpus]

    def __load_dataset(self, dataset_path):
        if not os.path.isfile(dataset_path):
            return [], []
        with open(dataset_path, 'r') as file_hanlder:
            raw_dataset = json.load(file_hanlder)

            dataset = [QARow(item['corrected_question'],
                             item['annotation'] if 'annotation' in item else '',
                             item['sparql_query'])
                       for item in
                       raw_dataset]
            dataset = [row for row in dataset if len(row.sparql.relations) == 1]
            corpus = [item.normalized_question for item in dataset]
            return dataset, corpus

    def __build_vocab(self, lines, vocab_path):
        vocab = set()
        for tokens in lines:
            vocab |= set(tokens)
        with open(vocab_path, 'w') as f:
            for token in sorted(vocab):
                f.write(token + '\n')
