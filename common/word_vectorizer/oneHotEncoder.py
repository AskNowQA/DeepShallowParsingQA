from common.word_vectorizer.wordVectorizer import WordVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class OneHotEncoder(WordVectorizer):
    def __init__(self, corpus):
        super(OneHotEncoder, self).__init__(corpus)
        self.vectorizer = CountVectorizer()
        y = self.vectorizer.fit_transform(self.corpus)
        self.word_size = y[0].shape[1]

    def decode(self, word_seq):
        output = []
        for word in word_seq.split():
            output.append(self.vectorizer.transform([word]).todense()[0])

        return np.array(output).reshape(len(output), -1)


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
        'Is this the first document?',
    ]

    vectorizer = OneHotEncoder(corpus)
    print(vectorizer.decode('is third'))
