class ElasticLinker:
    def __init__(self, elastic, index_name):
        self.elastic = elastic
        self.index_name = index_name

    def generate(self, surface, question, size=100):
        return self.elastic.search_index(surface, self.index_name, size=size)
