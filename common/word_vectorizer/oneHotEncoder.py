from common.word_vectorizer.wordVectorizer import WordVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class OneHotEncoder(WordVectorizer):
    def __init__(self, dataset):
        super(OneHotEncoder, self).__init__(dataset)
        self.vectorizer = CountVectorizer()
        y = self.vectorizer.fit_transform(self.dataset.corpus)
        self.word_size = y[0].shape[1]

    def decode(self, word_seq):
        output = []
        for word in word_seq.split():
            output.append(self.vectorizer.transform([word]).todense()[0])

        return np.array(output).reshape(len(output), -1)
