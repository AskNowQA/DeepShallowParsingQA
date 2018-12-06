class NGramLinker:
    def __init__(self, elastic, index_name):
        self.elastic = elastic
        self.index_name = index_name

    def generate(self, surface, question, size=10):
        return self.elastic.search_ngram(surface, self.index_name, size=size)
