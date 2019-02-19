from common.utils import *


class DatasetCG:
    def __init__(self, dataset, entity=False, relation=False):
        self.idx = 0
        if entity:
            self.idx = 0
        elif relation:
            self.idx = 1
        self.dataset = {item.question: [
            [[entity.raw_uri, entity.label] for entity in item.sparql.entities],
            [[relation.raw_uri, relation.label, relation.coded[0], relation.coded[1]] for relation in
             item.sparql.relations]] for item in
            dataset.train_set + dataset.test_set}

    @profile
    def generate(self, surface, question):
        if len(surface) > 2 and question in self.dataset:
            return self.dataset[question][self.idx]
        return []
