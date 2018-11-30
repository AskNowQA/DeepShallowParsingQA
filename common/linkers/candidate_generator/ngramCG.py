class NGramLinker:
    def __init__(self, elastic):
        self.elastic = elastic

    def generate(self, surface, question):
        return self.elastic.search_ngram(surface, 'idx')
