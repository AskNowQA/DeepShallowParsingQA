from common.dataset.base_dataset import Base_Dataset
from common.dataset.container.qarow import QARow
import ujson as json
import os


class SimpleDBpediaQA(Base_Dataset):
    def __init__(self, trainset_path, testset_path, vocab_path, remove_entity_mention=False, remove_stop_words=False):
        super(SimpleDBpediaQA, self).__init__(trainset_path, testset_path, vocab_path, 'SimpleDBpediaQA',
                                              remove_entity_mention,
                                              remove_stop_words)

    def load_dataset(self, dataset_path, remove_entity_mention, remove_stop_words):
        if not os.path.isfile(dataset_path):
            return [], []
        with open(dataset_path, 'r', encoding='utf-8') as file_hanlder:
            raw_dataset = json.load(file_hanlder)
            raw_dataset = raw_dataset['Questions']
            dataset = [QARow(item['Query'],
                             '',
                             'SELECT * WHERE {{<{}> <{}> ?x'.format(item['Subject'],
                                                                    item['PredicateList'][0]['Predicate']) +
                             ('' if item['PredicateList'][0]['Constraint'] is None else '. ?x xsd#type <{}>'.format(
                                 item['PredicateList'][0]['Constraint']) + '}'),
                             remove_entity_mention, remove_stop_words)
                       for item in
                       raw_dataset]
            corpus = [item.normalized_question for item in dataset]
            return dataset, corpus
