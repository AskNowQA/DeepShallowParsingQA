import pickle as pk
import ujson as json
from common.utils import *


class UnorderedLinker:
    def __init__(self, rel2id_path, core_chains_path, dataset):
        self.dataset = dataset
        with open(rel2id_path, 'rb') as f_h:
            self.rel2id = pk.load(f_h, encoding='latin1')
            self.id2rel = {v[0]: ([k] + v[1:]) for k, v in self.rel2id.items()}
        with open(core_chains_path, 'rb') as f_h:
            tmp = json.load(f_h)
            self.core_chains = {item['parsed-data']['corrected_question']: item for item in tmp}

    @profile
    def __find_core_chain(self, question):
        if question in self.core_chains:
            return self.core_chains[question]
        return None

    @profile
    def link(self, surface, question):
        core_chain = self.__find_core_chain(question)
        if core_chain is not None:
            hop_1 = core_chain['uri']['hop-1-properties']
            hop_2 = core_chain['uri']['hop-2-properties']
            hop_1 = [item[1] for item in hop_1]
            hop_2 = list(set([item[3] for item in hop_2]))
            results = [self.id2rel[id] for id in set(hop_1 + hop_2)]
            return results

    @profile
    def link_all(self, surfaces, qarow):
        output = []
        for surface in surfaces:
            output.append(self.link(surface, qarow.question))
        return output
