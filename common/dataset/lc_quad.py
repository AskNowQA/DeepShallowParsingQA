from common.dataset.container.qarow import QARow
from common.vocab import Vocab

import ujson as json
import os
import re


class LC_QuAD:
    def __init__(self, dataset_path, vocab_path):
        with open(dataset_path, 'r') as file_hanlder:
            self.raw_dataset = json.load(file_hanlder)

            self.dataset = [QARow(item['corrected_question'],
                                  item['annotation'] if 'annotation' in item else '',
                                  item['sparql_query'])
                            for item in
                            self.raw_dataset if len(re.findall('<[^>]*>', item['sparql_query'])) <= 2]
            self.corpus = [item.normalized_question for item in self.dataset]
            if not os.path.isfile(vocab_path):
                self.__build_vocab(self.corpus, vocab_path)
            self.vocab = Vocab(filename=vocab_path)
            self.coded_corpus = [[self.vocab.getIndex(word) for word in item.split()] for item in self.corpus]

    def __build_vocab(self, lines, vocab_path):
        vocab = set()
        for line in lines:
            tokens = line.rstrip('\n').split(' ')
            vocab |= set(tokens)
        with open(vocab_path, 'w') as f:
            for token in sorted(vocab):
                f.write(token + '\n')
