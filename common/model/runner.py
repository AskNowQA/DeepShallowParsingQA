import torch
import numpy as np
from tqdm import tqdm
import similarity.levenshtein
import similarity.ngram
import jellyfish
import logging
import os

from config import config
from common.model.agent import Agent
from common.model.policy import Policy
from common.model.policySplit import PolicySplit
from common.model.environment import Environment
from common.linkers.relationOrderLinker import RelationOrderedLinker
from common.linkers.entityOrderedLinker import EntityOrderedLinker
from common.linkers.sorter.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.sorter.embeddingSimilaritySorter import EmbeddingSimilaritySorter
from common.linkers.candidate_generator.graphCG import GraphCG
from common.linkers.candidate_generator.elasticLinker import ElasticLinker
from common.linkers.candidate_generator.datasetCG import DatasetCG
from common.linkers.candidate_generator.elastic import Elastic
from common.utils import *


class Runner:
    def __init__(self, lc_quad, args):
        self.logger = logging.getLogger('main')
        self.word_vectorizer = lc_quad.word_vectorizer
        self.elastic = Elastic(config['elastic']['server'])
        # string_similarity_metric = similarity.ngram.NGram(2).distance
        # string_similarity_metric = similarity.levenshtein.Levenshtein().distance
        # string_similarity_metric = jellyfish.levenshtein_distance
        entity_linker = EntityOrderedLinker(
            candidate_generator=DatasetCG(lc_quad, entity=True),
            sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, return_similarity=True)],
            vocab=lc_quad.vocab)

        relation_linker = RelationOrderedLinker(
            candidate_generator=GraphCG(rel2id_path=config['lc_quad']['rel2id'],
                                        core_chains_path=config['lc_quad']['core_chains'],
                                        dataset=lc_quad),
            sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, return_similarity=True),
                     EmbeddingSimilaritySorter(self.word_vectorizer)],
            vocab=lc_quad.vocab)

        policy_network = Policy(vocab_size=lc_quad.vocab.size(),
                                emb_size=self.word_vectorizer.word_size,
                                input_size=(self.word_vectorizer.word_size + 1) * 3 + 1 + 1,
                                hidden_size=self.word_vectorizer.word_size,
                                output_size=3,
                                dropout_ratio=args.dropout)
        policy_network.emb.weight.data.copy_(self.word_vectorizer.emb)
        split_network = PolicySplit(vocab_size=lc_quad.vocab.size(),
                                    emb_size=self.word_vectorizer.word_size,
                                    input_size=(self.word_vectorizer.word_size + 1) * 3 + 1 + 1,
                                    hidden_size=self.word_vectorizer.word_size,
                                    output_size=1,
                                    dropout_ratio=args.dropout)
        split_network.emb.weight.data.copy_(self.word_vectorizer.emb)
        self.agent = Agent(number_of_relations=2,
                           gamma=args.gamma,
                           policy_network=policy_network,
                           split_network=split_network,
                           policy_optimizer=torch.optim.Adam(
                               filter(lambda p: p.requires_grad, policy_network.parameters()), lr=args.lr),
                           split_optimizer=torch.optim.Adam(
                               filter(lambda p: p.requires_grad, split_network.parameters()), lr=args.lr))

        self.environment = Environment(entity_linker=entity_linker,
                                       relation_linker=relation_linker,
                                       positive_reward=args.positive_reward,
                                       negative_reward=args.negative_reward)

    def load_checkpoint(self, checkpoint_filename=config['checkpoint_path']):
        if os.path.isfile(checkpoint_filename):
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_filename)
            else:
                checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
            self.agent.policy_network.load_state_dict(checkpoint['model'])

    def save_checkpoint(self, checkpoint_filename=config['checkpoint_path']):
        checkpoint = {'model': self.agent.policy_network.state_dict()}
        torch.save(checkpoint, checkpoint_filename)

    @profile
    def train(self, lc_quad, args, checkpoint_filename=config['checkpoint_path']):
        total_reward, total_relation_rmm, total_entity_rmm, total_loss = [], [], [], []
        max_rmm, max_rmm_index = 0, -1
        iter = tqdm(range(args.epochs))
        history = {' '.join(qarow.normalized_question): [] for qarow in lc_quad.train_set}
        self.agent.policy_network.zero_grad()
        e = args.e
        for epoch in iter:
            for idx, qarow in enumerate(lc_quad.train_set):
                reward, relation_mrr, entity_mrr, loss, actions = self.step(
                    lc_quad.coded_train_corpus[idx],
                    qarow.lower_indicator,
                    qarow,
                    e=e,
                    k=args.k,
                    train=True)
                total_reward.append(reward)
                total_entity_rmm.append(entity_mrr)
                total_relation_rmm.append(relation_mrr)
                total_loss.append(loss)
                history[' '.join(qarow.normalized_question)].append(
                    actions.__str__() + '{:0.2f},{:0.2f},{:0.2f}'.format(entity_mrr, relation_mrr, reward))
                if idx % args.batchsize == 0:
                    self.agent.policy_optimizer.step()
                    self.agent.policy_network.zero_grad()

                    self.agent.split_optimizer.step()
                    self.agent.split_network.zero_grad()

            self.agent.policy_optimizer.step()
            self.agent.policy_network.zero_grad()
            self.agent.split_optimizer.step()
            self.agent.split_network.zero_grad()

            if epoch > 0 and epoch % 10 == 0:
                e = max(e - 0.001, 0.1)
                mean_rmm = [np.mean(total_entity_rmm), np.mean(total_relation_rmm)]
                print(list(map('{:0.2f}'.format, [np.mean(total_reward), np.mean(total_loss)] + mean_rmm)))
                total_reward, total_relation_rmm, total_entity_rmm, total_loss = [], [], [], []
                self.save_checkpoint(checkpoint_filename)
                if sum(mean_rmm) > max_rmm:
                    max_rmm = sum(mean_rmm)
                    max_rmm_index = epoch
                # else:
                #     if epoch >= max_rmm_index + 30:
                #         iter.close()•
                #         break
        if len(total_reward) > 0:
            print(list(map('{:0.2f}'.format, [np.mean(total_reward), np.mean(total_loss), np.mean(total_entity_rmm),
                                              np.mean(total_relation_rmm)])))

    def test(self, lc_quad, args):
        self.environment.entity_linker = EntityOrderedLinker(
            candidate_generator=ElasticLinker(self.elastic, index_name='entity_whole_match_index'),
            sorters=[StringSimilaritySorter(similarity.ngram.NGram(2).distance)],
            vocab=lc_quad.vocab)

        self.environment.relation_linker = RelationOrderedLinker(
            # candidate_generator=GraphCG(rel2id_path=config['lc_quad']['rel2id'],
            #                             core_chains_path=config['lc_quad']['core_chains'],
            #                             dataset=lc_quad),
            candidate_generator=ElasticLinker(self.elastic, index_name='relation_whole_match_index'),
            sorters=[StringSimilaritySorter(jellyfish.levenshtein_distance, return_similarity=True),
                     ],  # EmbeddingSimilaritySorter(self.word_vectorizer)],
            vocab=lc_quad.vocab)
        total_relation_mrr, total_entity_mrr = [], []
        # for idx, qarow in enumerate(lc_quad.train_set):
        #     reward, relation_mrr, entity_mrr, loss, _ = self.step(
        #         lc_quad.coded_train_corpus[idx],
        for idx, qarow in enumerate(lc_quad.test_set):
            reward, relation_mrr, entity_mrr, loss, _ = self.step(
                lc_quad.coded_test_corpus[idx],
                qarow.lower_indicator, qarow,
                e=args.e,
                train=False,
                k=args.k)
            total_relation_mrr.append(relation_mrr)
            total_entity_mrr.append(entity_mrr)

        total_entity_mrr = np.mean(total_entity_mrr)
        total_relation_mrr = np.mean(total_relation_mrr)
        print('entity MRR', total_entity_mrr)
        print('relation MRR', total_relation_mrr)
        return total_entity_mrr, total_relation_mrr

    @profile
    def step(self, input, lower_indicator, qarow, e, train=True, k=0):
        rewards, action_log_probs, action_probs, actions, split_actions = [], [], [], [], []
        loss = 0
        running_reward = 0
        self.environment.init(input, lower_indicator)
        self.agent.init()
        state = self.environment.state
        while True:
            action, action_log_prob, action_prob, split_action = self.agent.select_action(state, e, train)
            split_actions.append(split_action)
            actions.append(int(action))
            action_log_probs.append(action_log_prob)
            action_probs.append(action_prob)
            new_state, detailed_rewards, total_reward, split_action_target, done, relation_mrr, entity_mrr = \
                self.environment.step(action, action_probs, int(split_action >= 0.5), qarow, k, train=train)
            running_reward += total_reward
            # rewards.append(total_reward)
            state = new_state
            if done:
                if train:
                    loss, split_loss = self.agent.backward(detailed_rewards, total_reward, action_log_probs,
                                                           split_actions, split_action_target)
                break
            running_reward = min(running_reward, 1)
        del action_log_prob
        return running_reward, relation_mrr, entity_mrr, loss, actions
