from common.vocab import Vocab

import ujson as json
import os


class LC_QuAD:
    def __init__(self, dataset_path, vocab_path):
        with open(dataset_path, 'r') as file_hanlder:
            self.raw_dataset = json.load(file_hanlder)

            self.dataset = [[self.__preprocess(item['corrected_question']),
                             item['corrected_question'],
                             item['annotation']] for item in
                            self.raw_dataset]
            self.corpus = [item[0] for item in self.dataset]
            self.validate = all([(len(item[0].split()) == len(item[2])) for item in self.dataset])
            if self.validate:
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

    def __preprocess(self, line):
        return line.lower().replace('?', ' ').replace('\'', ' ').replace('-', ' ').replace(',', ' ')
