import re
from common.utils import *


class URI:
    def __init__(self, raw_uri, ngram=0):
        self.raw_uri = raw_uri.strip('<>')
        self.label = self.raw_uri[self.raw_uri.rindex('/') + 1:]
        # self.tokens = [token.replace('.', '') for token in re.sub(r"([A-Z])", r" \1", self.label).split()]
        self.tokens = URI.normalize(self.label)
        self.label = self.label.replace('_', ' ').lower()
        self.coded = ([], 0)
        self.ngram = None
        if ngram > 0:
            self.ngram = Utils.ngrams(self.label, ngram)

    @staticmethod
    def normalize(word):
        return re.sub(r"\s+", ' ',
                      re.sub(r"[^\w]|\d+|_", ' ', re.sub(r"([A-Z])", r" \1", word)).strip()).lower().split()
