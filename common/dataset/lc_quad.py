from common.dataset.base_dataset import Base_Dataset
from common.dataset.container.qarow import QARow
import ujson as json
import os


class LC_QuAD(Base_Dataset):
    def __init__(self, trainset_path, testset_path, vocab_path, remove_entity_mention=False, remove_stop_words=False):
        super(LC_QuAD, self).__init__(trainset_path, testset_path, vocab_path, 'lc_quad', remove_entity_mention,
                                      remove_stop_words)
        # self.train_set, self.train_corpus = self.__load_dataset(trainset_path, remove_entity_mention, remove_stop_words)
        # self.test_set, self.test_corpus = self.__load_dataset(testset_path, remove_entity_mention, remove_stop_words)
        #
        # self.corpus = self.train_corpus + self.test_corpus
        # if not os.path.isfile(vocab_path):
        #     self.__build_vocab(self.corpus, vocab_path)
        # self.vocab = Vocab(filename=vocab_path, data=['<ent>', '<num>'])
        # self.word_vectorizer = Glove(self.vocab, config['glove_path'], config['lc_quad']['emb'])
        # # self.__update_relations_emb()
        #
        # self.coded_train_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.train_corpus]
        # self.coded_test_corpus = [[self.vocab.getIndex(word) for word in tokens] for tokens in self.test_corpus]
        # self.vocab_path = vocab_path
        #
        # self.one_hop = None
        # if os.path.isfile(config['lc_quad']['entity_one_hop']):
        #     with open(config['lc_quad']['entity_one_hop'], 'rb') as f:
        #         self.one_hop = pk.load(f)

    def load_dataset(self, dataset_path, remove_entity_mention, remove_stop_words):
        if not os.path.isfile(dataset_path):
            return [], []
        with open(dataset_path, 'r', encoding='utf-8') as file_hanlder:
            raw_dataset = json.load(file_hanlder)
            dataset = [QARow(item['corrected_question'],
                             item['annotation'] if 'annotation' in item else '',
                             item['sparql_query'],
                             remove_entity_mention, remove_stop_words)
                       for item in
                       raw_dataset]
            # with open('/Users/hamid/workspace/DeepShallowParsingQA/data/lcquad/no_constraints.json', 'r') as f:
            #     no_contraints = json.load(f)
            #     dataset = [row for row in dataset if row.question in no_contraints]
            # dataset = [row for row in dataset if  (len(row.sparql.relations) == 1 and len(row.sparql.entities) == 1)]
            # dataset = [row for row in dataset if len(row.normalized_question) == 3]
            # dataset = dataset[:10]
            corpus = [item.normalized_question for item in dataset]
            return dataset, corpus
