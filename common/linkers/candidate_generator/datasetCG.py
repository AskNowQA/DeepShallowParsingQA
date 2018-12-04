from common.utils import *


class DatasetCG:
    def __init__(self, dataset):
        self.dataset = {item.question: [[entity.raw_uri, entity.label] for entity in item.sparql.entities] for item in
                        dataset.train_set + dataset.test_set}

    @profile
    def generate(self, surface, question):
        if question in self.dataset:
            return self.dataset[question]
        return []
