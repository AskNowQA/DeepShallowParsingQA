from common.word_vectorizer.wordVectorizer import WordVectorizer
from common.vocab import Vocab
import torch
import os
import numpy as np
import ujson as json


class Glove(WordVectorizer):
    def __init__(self, corpus, path):
        super(Glove, self).__init__(corpus)

        if os.path.isfile(path + '.emd') and os.path.isfile(path + '.corpus'):
            with open(path + '.corpus', 'r') as file_handler:
                self.doc_idx = json.load(file_handler)
            self.embd = torch.load(path + '.emd')
            self.vocab = Vocab()
            self.vectors = []
            self.word_size = self.embd[0].size(1)
        else:
            self.vocab, self.vectors = self.load_word_vectors(path)
            self.word_size = self.vectors.size(1)
            self.doc_idx = {}
            self.embd = [self.decode(doc) for doc in corpus]
            self.doc_idx = {doc: idx for idx, doc in enumerate(corpus)}
            with open(path + '.corpus', 'w') as file_handler:
                json.dump(self.doc_idx, file_handler)
            torch.save(self.embd, path + '.emd')

    def load_word_vectors(self, path):
        """
        loading GLOVE word vectors
            if .pth file is found, will load that
            else will load from .txt file & save
        :param path:
        :return:
        """
        if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
            print('==> File found, loading to memory')
            vectors = torch.load(path + '.pth')
            vocab = Vocab(filename=path + '.vocab')
            return vocab, vectors
        # saved file not found, read from txt file
        # and create tensors for word vectors
        print('==> File not found, preparing, be patient')
        count = sum(1 for line in open(path + '.txt'))
        with open(path + '.txt', 'r') as f:
            contents = f.readline().rstrip('\n').split(' ')
            dim = len(contents[1:])
        words = [None] * (count)
        vectors = torch.zeros(count, dim)
        with open(path + '.txt', 'r') as f:
            idx = 0
            for line in f:
                contents = line.rstrip('\n').split(' ')
                words[idx] = contents[0]
                vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
                idx += 1
        with open(path + '.vocab', 'w') as f:
            for word in words:
                f.write(word + '\n')
        vocab = Vocab(filename=path + '.vocab')
        torch.save(vectors, path + '.pth')
        return vocab, vectors

    def decode(self, word_seq):
        if word_seq in self.doc_idx:
            return self.embd[self.doc_idx[word_seq]]

        word_seq = word_seq.split()
        output = torch.Tensor(len(word_seq), self.word_size).normal_(-0.05, 0.05)
        for idx, word in enumerate(word_seq):
            if word in self.vocab.labelToIdx:
                output[idx] = self.vectors[self.vocab.labelToIdx[word]]

        return output


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
        'Is this the first document?',
    ]

    vectorizer = Glove(corpus, '/Users/hamid/workspace/SQG/learning/treelstm/data/glove/glove.840B.300d')
    print(vectorizer.decode('this is the third one.'))
