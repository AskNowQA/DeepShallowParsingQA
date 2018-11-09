import torch
import numpy as np
from tqdm import tqdm

from config import config
from common.word_vectorizer.glove import Glove
from common.model.agent import Agent
from common.model.policy import Policy
from common.model.environment import Environment
from common.linkers.orderedLinker import OrderedLinker
from common.linkers.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.embeddingSimilaritySorter import EmbeddingSimilaritySorter


class Runner:
    def __init__(self, lc_quad, args):
        word_vectorizer = Glove(lc_quad, config['glove_path'], config['lc_quad']['emb'])
        policy_network = Policy(vocab_size=lc_quad.vocab.size(),
                                emb_size=word_vectorizer.word_size,
                                input_size=word_vectorizer.word_size * 2 + 1,
                                hidden_size=int(word_vectorizer.word_size * 4),
                                output_size=2,
                                dropout_ratio=args.dropout)
        policy_network.emb.weight.data.copy_(word_vectorizer.emb)
        self.agent = Agent(number_of_relations=2,
                           gamma=args.gamma,
                           policy_network=policy_network,
                           policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=args.lr))
        if args.sim == 'str':
            sorter = StringSimilaritySorter()
        else:
            sorter = EmbeddingSimilaritySorter(word_vectorizer)
        linker = OrderedLinker(sorter=sorter,
                               rel2id_path=config['lc_quad']['rel2id'],
                               core_chains_path=config['lc_quad']['core_chains'],
                               dataset=lc_quad)
        self.environment = Environment(linker=linker,
                                       positive_reward=args.positive_reward,
                                       negative_reward=args.negative_reward)

    def load_checkpoint(self, checkpoint_filename=config['checkpoint_path']):
        checkpoint = torch.load(checkpoint_filename)
        self.agent.policy_network.load_state_dict(checkpoint['model'])

    def save_checkpoint(self, checkpoint_filename=config['checkpoint_path']):
        checkpoint = {'model': self.agent.policy_network.state_dict()}
        torch.save(checkpoint, checkpoint_filename)

    def train(self, lc_quad, args, checkpoint_filename=config['checkpoint_path']):
        total_reward, total_rmm, total_loss = [], [], []
        min_loss, min_loss_index = 100, -1
        last_idx = 0
        iter = tqdm(range(args.epochs))
        for epoch in iter:
            for idx, qarow in enumerate(lc_quad.train_set):
                reward, mrr, loss = self.step(lc_quad.coded_train_corpus[idx], qarow, e=args.e, k=args.k)
                total_reward.append(reward)
                total_rmm.append(mrr)
                total_loss.append(float(loss))
            if epoch > 0 and epoch % 10 == 0:
                mean_loss = np.mean(total_loss[last_idx:])
                print(np.mean(total_reward[last_idx:]), np.mean(total_rmm[last_idx:]), mean_loss)
                last_idx = len(total_reward)
                self.save_checkpoint(checkpoint_filename)
                if mean_loss <= min_loss:
                    min_loss = mean_loss
                    min_loss_index = epoch
                else:
                    if epoch >= min_loss_index + 3:
                        iter.close()
                        break
        if len(total_reward[last_idx:]) > 0:
            print(np.mean(total_reward[last_idx:]), np.mean(total_rmm[last_idx:]), np.mean(total_loss[last_idx:]))

    def test(self, lc_quad, args):
        total_rmm = []
        for idx, qarow in enumerate(lc_quad.test_set):
            reward, mrr, loss = self.step(lc_quad.coded_test_corpus[idx], qarow, e=args.e, train=False, k=args.k)
            total_rmm.append(mrr)
        total = np.mean(total_rmm)
        print(total)
        return total

    def step(self, input, qarow, e, train=True, k=0):
        rewards, action_log_probs, total_reward = [], [], []
        loss = 0
        running_reward = 0
        self.environment.init(input)
        state = self.environment.state
        while True:
            action_dist, action, action_log_prob = self.agent.select_action(state, e)
            new_state, reward, done, mrr = self.environment.step(action, qarow, train, k)
            running_reward += reward
            rewards.append(reward)
            action_log_probs.append(action_log_prob)
            state = new_state
            if done:
                if train:
                    loss = self.agent.optimize(rewards, action_log_probs)
                total_reward.append(running_reward)
                break
        return total_reward, mrr, loss
