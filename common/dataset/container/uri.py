import re
from common.utils import *


class URI:
    def __init__(self, raw_uri, ngram=0):
        self.raw_uri = raw_uri.strip('<>')
        self.label = self.raw_uri[self.raw_uri.rindex('/') + 1:]
        self.tokens = re.sub(r"([A-Z])", r" \1", self.label).split()
        self.label = self.label.replace('_', ' ').lower()
        self.ngram = None
        if ngram > 0:
            self.ngram = Utils.ngrams(self.label, ngram)
