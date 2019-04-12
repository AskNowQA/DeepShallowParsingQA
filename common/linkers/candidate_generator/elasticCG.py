from config import config
import pickle as pk
import torch


class ElasticCG:
    def __init__(self, elastic, index_name):
        self.elastic = elastic
        self.index_name = index_name

        with open(config['dbpedia']['relations'] + '.coded', 'rb') as file_handler:
            self.coded_uris = pk.load(file_handler)

    def extract_info(self, uri):
        label = uri[uri.rindex('/') + 1:]
        tokens = torch.LongTensor([0, 0, 0])
        length = 0
        if uri in self.coded_uris:
            tokens, length = self.coded_uris[uri]

        return [uri, label, tokens, length]

    def generate(self, surfaces, extra_surfaces, surface, question, size=100):
        output = self.elastic.search_index(surface, self.index_name, size=size)

        if 'relation' in self.index_name:
            output = set([item[0] for item in output])
            output = [self.extract_info(item) for item in output]

        return output
