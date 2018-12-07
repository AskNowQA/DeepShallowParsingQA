import torch
import numpy as np
from tqdm import tqdm
import similarity.levenshtein
import similarity.ngram
import jellyfish
import logging

from config import config
from common.model.agent import Agent
from common.model.policy import Policy
from common.model.environment import Environment
from common.linkers.relationOrderLinker import RelationOrderedLinker
from common.linkers.entityOrderedLinker import EntityOrderedLinker
from common.linkers.sorter.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.sorter.embeddingSimilaritySorter import EmbeddingSimilaritySorter
from common.linkers.candidate_generator.graphCG import GraphCG
from common.linkers.candidate_generator.ngramCG import NGramLinker
from common.linkers.candidate_generator.datasetCG import DatasetCG
from common.linkers.candidate_generator.elastic import Elastic
from common.utils import *


class Runner:
    def __init__(self, lc_quad, args):
        self.logger = logging.getLogger('main')
        word_vectorizer = lc_quad.word_vectorizer
        self.elastic = Elastic(config['elastic']['server'],
                               config['elastic']['entity_ngram_index_config'],
                               config['dbpedia']['entities'],
                               index_name='entity_whole_match',
                               create_entity_index=False)
        # string_similarity_metric = similarity.ngram.NGram(2).distance
        # string_similarity_metric = similarity.levenshtein.Levenshtein().distance
        string_similarity_metric = jellyfish.levenshtein_distance
        entity_linker = EntityOrderedLinker(
            candidate_generator=DatasetCG(lc_quad),
            sorters=[StringSimilaritySorter(string_similarity_metric)],
            vocab=lc_quad.vocab)

        relation_linker = RelationOrderedLinker(
            candidate_generator=GraphCG(rel2id_path=config['lc_quad']['rel2id'],
                                        core_chains_path=config['lc_quad']['core_chains'],
                                        dataset=lc_quad),
            sorters=[StringSimilaritySorter(string_similarity_metric),
                     EmbeddingSimilaritySorter(word_vectorizer)],
            vocab=lc_quad.vocab)

        policy_network = Policy(vocab_size=lc_quad.vocab.size(),
                                emb_size=word_vectorizer.word_size,
                                input_size=word_vectorizer.word_size * 3 + 1 + 1,
                                hidden_size=word_vectorizer.word_size,
                                output_size=3,
                                dropout_ratio=args.dropout)
        policy_network.emb.weight.data.copy_(word_vectorizer.emb)
        self.agent = Agent(number_of_relations=2,
                           gamma=args.gamma,
                           policy_network=policy_network,
                           policy_optimizer=torch.optim.Adam(
                               filter(lambda p: p.requires_grad, policy_network.parameters()), lr=args.lr))

        self.environment = Environment(entity_linker=entity_linker,
                                       relation_linker=relation_linker,
                                       positive_reward=args.positive_reward,
                                       negative_reward=args.negative_reward)

    def load_checkpoint(self, checkpoint_filename=config['checkpoint_path']):
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
        total_reward, total_rmm, total_loss = [], [], []
        max_rmm, max_rmm_index = 0, -1
        iter = tqdm(range(args.epochs))
        self.agent.policy_network.zero_grad()
        for epoch in iter:
            for idx, qarow in enumerate(lc_quad.train_set):
                reward, mrr, loss = self.step(lc_quad.coded_train_corpus[idx], qarow, e=args.e, k=args.k, train=True)
                total_reward.append(reward)
                total_rmm.append(mrr)
                total_loss.append(loss)
                if idx % args.batchsize == 0:
                    self.agent.policy_optimizer.step()
                    self.agent.policy_network.zero_grad()

            self.agent.policy_optimizer.step()
            self.agent.policy_network.zero_grad()

            if epoch > 0 and epoch % 10 == 0:
                mean_rmm = np.mean(total_rmm)
                print(np.mean(total_reward), mean_rmm, np.mean(total_loss))
                total_reward, total_rmm, total_loss = [], [], []
                self.save_checkpoint(checkpoint_filename)
                if mean_rmm > max_rmm:
                    max_rmm = mean_rmm
                    max_rmm_index = epoch
                else:
                    if epoch >= max_rmm_index + 30:
                        iter.close()
                        break
        if len(total_reward) > 0:
            print(np.mean(total_reward), np.mean(total_rmm), np.mean(total_loss))

    def test(self, lc_quad, args):
        self.environment.entity_linker.candidate_generator = NGramLinker(self.elastic, index_name='entity_whole_match')
        self.environment.entity_linker.sorters = [StringSimilaritySorter(similarity.ngram.NGram(2).distance)]
        try_total = []
        for i in range(5):
            total_rmm = []
            for idx, qarow in enumerate(lc_quad.test_set):
                reward, mrr, loss = self.step(lc_quad.coded_test_corpus[idx], qarow, e=args.e, train=False, k=args.k)
                total_rmm.append(mrr)
            total = np.mean(total_rmm)
            print(total)
            try_total.append(total)
        print(np.mean(try_total), np.min(try_total), np.max(try_total))
        return total

    @profile
    def step(self, input, qarow, e, train=True, k=0):
        rewards, action_log_probs, action_probs, total_reward = [], [], [], []
        loss = 0
        running_reward = 0
        self.environment.init(input)
        state = self.environment.state
        while True:
            action, action_log_prob, action_prob = self.agent.select_action(state, e)
            action_log_probs.append(action_log_prob)
            action_probs.append(action_prob)
            new_state, reward, done, mrr = self.environment.step(action, action_probs, qarow, k, train=train)
            running_reward += reward
            rewards.append(reward)
            state = new_state
            if done:
                if train:
                    loss = self.agent.backward(rewards, action_log_probs)
                total_reward.append(running_reward)
                break
        del action_log_prob
        return total_reward, mrr, loss
