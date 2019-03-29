class ElasticCG:
    def __init__(self, elastic, index_name):
        self.elastic = elastic
        self.index_name = index_name

    def generate(self, surfaces, extra_surfaces, surface, question, size=100):
        output = self.elastic.search_index(surface, self.index_name, size=size)

        if 'relation' in self.index_name:
            output = set([item[0] for item in output])
            output = [[item, item[item.rindex('/') + 1:]] for item in output]

        return output
