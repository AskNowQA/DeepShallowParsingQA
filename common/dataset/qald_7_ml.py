from common.dataset.container.qarow import QARow
from common.dataset.base_dataset import Base_Dataset
import ujson as json
import os
import re


class Qald_7_ml(Base_Dataset):
    def __init__(self, trainset_path, testset_path, vocab_path, remove_entity_mention=False, remove_stop_words=False):
        super(Qald_7_ml, self).__init__(trainset_path, testset_path, vocab_path, 'qald_7_ml', remove_entity_mention,
                                        remove_stop_words)

    def __reformat_sparql(self, sparql):
        sparql_lower = sparql.lower()
        if 'select ' in sparql_lower:
            sparql = sparql[sparql_lower.index('select '):]
        elif 'ask ' in sparql_lower or 'ask\n' in sparql_lower:
            sparql = sparql[sparql_lower.index('ask'):]
        else:
            print('pita')
        sparql = sparql.replace('res:', '<http://dbpedia.org/resource/').replace(
            'dbr:', '<http://dbpedia.org/resource/').replace(
            'dbo:', '<http://dbpedia.org/ontology/')
        for item in re.findall('<[^ ]*', sparql):
            if item[-1] != '>':
                if item[-1] == '.':
                    item = item[:-1]
                sparql = sparql.replace(item, item + '>')
        return sparql

    def load_dataset(self, dataset_path, remove_entity_mention, remove_stop_words):
        if not os.path.isfile(dataset_path):
            return [], []
        with open(dataset_path, 'r') as file_hanlder:
            raw_dataset = json.load(file_hanlder)

            dataset = [QARow(item['question'][0]['string'],
                             item['annotation'] if 'annotation' in item else '',
                             self.__reformat_sparql(item['query']['sparql']),
                             remove_entity_mention, remove_stop_words)
                       for item in
                       raw_dataset['questions']]
            # dataset = dataset[:5]
            corpus = [item.normalized_question for item in dataset]
            return dataset, corpus
