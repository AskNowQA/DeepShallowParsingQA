import torch
import pickle as pk
import ujson as json
from common.utils import *


class GraphCG:
    def __init__(self, rel2id_path, core_chains_path, dataset):
        self.dataset = dataset
        with open(rel2id_path, 'rb') as f_h:
            self.rel2id = pk.load(f_h, encoding='latin1')
            self.id2rel = {v[0]: ([k] + v[1:]) for k, v in self.rel2id.items()}
        with open(core_chains_path, 'rb') as f_h:
            tmp = json.load(f_h)
            self.core_chains = {item['parsed-data']['corrected_question']: item for item in tmp}

    def __find_core_chain(self, question):
        if question in self.core_chains:
            return self.core_chains[question]
        return None

    @profile
    def generate(self, surfaces, extra_surfaces, surface, question):
        core_chain = self.__find_core_chain(question)
        if core_chain is not None:
            hop_1 = core_chain['uri']['hop-1-properties']
            hop_2 = core_chain['uri']['hop-2-properties']
            hop_1 = set([item[1] for item in hop_1])
            hop_2 = set([item[3] for item in hop_2])
            results = [self.id2rel[id] for id in hop_1 | hop_2]
            qa_row = [item for item in self.dataset.train_set if item.question == question]
            if len(qa_row) > 0:
                qa_row = qa_row[0]
                target_uri_not_in_results = [uri.raw_uri for uri in qa_row.sparql.relations if
                                             uri.raw_uri not in [item[0] for item in results]]

                if len(target_uri_not_in_results) > 0:
                    for item in target_uri_not_in_results:
                        results.append(
                            [item, item[item.rindex('/') + 1:], [], [], [], torch.zeros([3], dtype=torch.int64), 1])

            return results
