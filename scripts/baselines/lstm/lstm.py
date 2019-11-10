import jellyfish
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import similarity.ngram

from config import config
from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.qald_7_ml import Qald_7_ml
from common.linkers.candidate_generator.datasetCG import DatasetCG
from common.linkers.sorter.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.sorter.embeddingSimilaritySorter import EmbeddingSimilaritySorter
from common.model.environment import Environment
from common.linkers.entityOrderedLinker import EntityOrderedLinker
from common.linkers.relationOrderLinker import RelationOrderedLinker
from common.linkers.candidate_generator.elasticCG import ElasticCG
from common.linkers.candidate_generator.elastic import Elastic


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, input_size, hidden_size, output_size, dropout_ratio, emb_idx,
                 bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.output_size = output_size
        self.emb_idx = emb_idx

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0, sparse=False)
        self.emb.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout_ratio, bidirectional=bidirectional)
        self.hidden2action = nn.Linear(hidden_size + (bidirectional * hidden_size), output_size)
        self.activation = nn.Softmax(dim=2)
        self.hidden = None

    def init(self):
        self.hidden = None

    def forward(self, input, hidden=None):
        num_words = len(input)
        input = self.emb(input)
        if hidden is None:
            hidden = self.hidden
        if hidden is None:
            lstm_out, hidden = self.lstm(input.reshape(1, num_words, -1))
        else:
            lstm_out, hidden = self.lstm(input.reshape(1, num_words, -1), hidden)
        action_space = self.hidden2action(lstm_out)
        action_score = self.activation(action_space)

        self.hidden = hidden
        return action_score.reshape(num_words, -1)


def train(dataset, classifier, loss_function, optimizer, num_epoch=10):
    for epoch in tqdm(range(num_epoch)):  # again, normally you would NOT do 300 epochs, it is toy data
        loss_val = []
        for sentence, tags in dataset:
            classifier.zero_grad()
            classifier.init()
            sentence_in = sentence
            targets = tags

            # Step 3. Run our forward pass.
            tag_scores = classifier(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss_val.append(loss.data)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(np.mean(loss_val))


def eval(dataset, coded_dataset, classifier, entity_linker, relation_linker, loss_function, k=0):
    with torch.no_grad():
        # loss_val = []
        entity_mrrs = []
        relation_mrrs = []
        for idx, qarow in tqdm(enumerate(dataset.test_set)):
            inputs = torch.LongTensor(dataset.coded_test_corpus[idx])
            classifier.init()
            tag_scores = classifier(inputs)
            action_seq = [torch.distributions.Categorical(scores).sample() for scores in tag_scores]
            print(qarow.question)
            surfaces, splitted_relations = Environment.find_surfaces(qarow.normalized_question_with_numbers, action_seq,
                                                                     [1] * len(action_seq))
            print(surfaces)
            # print(list(map(int, action_seq)), inputs[1])
            # loss = loss_function(tag_scores, inputs[1])
            # loss_val.append(loss.data)

            extra_candidates = []
            entity_results, entity_score, entity_mrr, found_target_entities = entity_linker.best_ranks(
                list(surfaces[1]), list(surfaces[0]), qarow, k, False)
            extra_candidates.extend(dataset.find_one_hop_relations(found_target_entities))
            relation_results, relation_score, relation_mrr, _ = relation_linker.best_ranks(
                list(surfaces[0]), list(surfaces[1]), qarow, k, False, extra_candidates)
            print(entity_mrr, relation_mrr)
            entity_mrrs.append(entity_mrr)
            relation_mrrs.append(relation_mrr)

        # print(np.mean(loss_val))
        print(np.mean(entity_mrrs))
        print(np.mean(relation_mrrs))


if __name__ == '__main__':
    stop_words = set(stopwords.words('english') + ['whose']) - set(['where', 'when'])
    k = 10
    # ds_name = 'q6'
    # dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
    #                     False, False)
    # ds_name = 'q7'
    # dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
    #                     False, False)
    ds_name = 'lcquad'
    dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      False, False)
    word_vectorizer = dataset.word_vectorizer

    entity_cg = DatasetCG(dataset, entity=True)
    entity_sorters = [StringSimilaritySorter(jellyfish.levenshtein_distance, False, True)]

    relation_cg = DatasetCG(dataset, relation=True)
    relation_sorters = [StringSimilaritySorter(jellyfish.levenshtein_distance, False, True),
                        EmbeddingSimilaritySorter(word_vectorizer, 0.3)]

    train_set_file_name = 'prep-{}.json'.format(ds_name)
    if os.path.exists(train_set_file_name):
        train_set = torch.load(train_set_file_name)
    else:
        train_set = {}
        for idx, item in enumerate(dataset.coded_train_corpus):

            qarow = dataset.train_set[idx]

            used_relations = {}
            labels = []
            words = []
            for word in qarow.normalized_question:
                targets = entity_cg.generate('', '', word, qarow.question)
                entity_scores = [sorter.sort(word, qarow.question, targets) for sorter in entity_sorters]
                entity_scores = [item[0][-1] for item in entity_scores if len(item) > 0 and item[0][-1] > 0.1]

                targets = relation_cg.generate('', '', word, qarow.question)
                relation_scores = [sorter.sort(word, qarow.question, targets) for sorter in relation_sorters]
                # relation_scores = {item[0][0]: item[0][-1] for item in relation_scores if len(item) > 0 and item[0][-1] > 0.1}
                relation_scores = [item[0][-1] for item in relation_scores if len(item) > 0 and item[0][-1] > 0.1]

                entity_score = 0
                if len(entity_scores) > 0:
                    entity_score = np.mean(entity_scores)
                relation_score = 0
                if len(relation_scores) > 0:
                    relation_score = np.mean(relation_scores)
                if max(entity_score, relation_score) < 0.15:
                    label = 0
                else:
                    if entity_score > relation_score:
                        label = 2
                    else:
                        if word.lower() in stop_words:
                            label = 0
                        else:
                            label = 1
                word_coded = dataset.vocab.getIndex(word.lower())
                if word_coded is None:
                    print(word)

                words.append(torch.LongTensor([word_coded]))
                labels.append(label)
            train_set[qarow.question] = [torch.LongTensor(words), torch.LongTensor(labels)]
            # if idx > 5:
            #     break
            # break
        # for item in train_set:
        #     print(item, train_set[item])
        torch.save(train_set, train_set_file_name)

    classifier = LSTMClassifier(vocab_size=dataset.vocab.size(),
                                emb_size=dataset.word_vectorizer.word_size,
                                input_size=dataset.word_vectorizer.word_size,
                                hidden_size=dataset.word_vectorizer.word_size,
                                output_size=3,
                                dropout_ratio=0.1,
                                emb_idx=0,
                                bidirectional=False)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.1)

    checkpoint_filename = ds_name + '.chk'
    if os.path.isfile(checkpoint_filename):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_filename)
        else:
            checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(checkpoint['model'])
    else:
        train(train_set.values(), classifier, loss_function, optimizer, 100)
        checkpoint = {'model': classifier.state_dict()}
        torch.save(checkpoint, checkpoint_filename)

    elastic = Elastic(config['elastic']['server'])
    entity_linker = EntityOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='entity_whole_match_index'),
        sorters=[StringSimilaritySorter(similarity.ngram.NGram(2).distance, True)],
        vocab=dataset.vocab)
    relation_linker = RelationOrderedLinker(
        candidate_generator=ElasticCG(elastic, index_name='relation_whole_match_index'),
        sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, False, True),
                 EmbeddingSimilaritySorter(dataset.word_vectorizer)],
        vocab=dataset.vocab)
    eval(dataset, train_set, classifier, entity_linker, relation_linker, loss_function)

## lstm train
# 0.47689583333333335
# 0.2860258928571428

## q6 train
# 0.4353233830845771
# 0.2018407960199005
