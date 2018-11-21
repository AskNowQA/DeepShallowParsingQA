import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data

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
        joint_vocab = lc_quad.vocab
        joint_vocab.loadFile(config['lc_quad']['rel_vocab'])
        word_vectorizer = Glove(joint_vocab, config['glove_path'], config['lc_quad']['emb'])
        linker = OrderedLinker(sorters=[StringSimilaritySorter(), EmbeddingSimilaritySorter(word_vectorizer)],
                               rel2id_path=config['lc_quad']['rel2id'],
                               core_chains_path=config['lc_quad']['core_chains'],
                               dataset=lc_quad)

        policy_network = Policy(vocab_size=lc_quad.vocab.size(),
                                emb_size=word_vectorizer.word_size,
                                input_size=word_vectorizer.word_size * 3 + 1 + 1,
                                hidden_size=word_vectorizer.word_size,
                                output_size=2,
                                dropout_ratio=args.dropout)
        policy_network.emb.weight.data.copy_(word_vectorizer.emb)
        self.agent = Agent(number_of_relations=2,
                           gamma=args.gamma,
                           policy_network=policy_network,
                           policy_optimizer=torch.optim.Adam(
                               filter(lambda p: p.requires_grad, policy_network.parameters()), lr=args.lr))

        self.environment = Environment(linker=linker,
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

    def train(self, lc_quad, args, checkpoint_filename=config['checkpoint_path']):
        total_reward, total_rmm, total_loss = [], [], []
        max_rmm, max_rmm_index = 0, -1
        last_idx = 0
        iter = tqdm(range(args.epochs))
        self.agent.policy_network.zero_grad()
        batch = torch.utils.data.DataLoader(lc_quad.dataset__, batch_size=len(lc_quad.train_set))#, pin_memory=True)
        batch = list(batch)[0][0]
        for epoch in iter:
            for idx, datarow in enumerate(batch):
                qarow = lc_quad.train_set[idx]
                reward, mrr, loss = self.step(datarow, qarow, e=args.e, k=args.k) # lc_quad.coded_train_corpus
                total_reward.append(reward)
                total_rmm.append(mrr)
                total_loss.append(float(loss))
                if idx % args.batchsize == 0:
                    self.agent.policy_optimizer.step()
                    self.agent.policy_network.zero_grad()

            self.agent.policy_optimizer.step()
            self.agent.policy_network.zero_grad()

            if epoch > 0 and epoch % 10 == 0:
                mean_rmm = np.mean(total_rmm[last_idx:])
                print(np.mean(total_reward[last_idx:]), mean_rmm, np.mean(total_loss[last_idx:]))
                last_idx = len(total_reward)
                self.save_checkpoint(checkpoint_filename)
                if mean_rmm > max_rmm:
                    max_rmm = mean_rmm
                    max_rmm_index = epoch
                else:
                    if epoch >= max_rmm_index + 30:
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
                    loss = self.agent.backward(rewards, action_log_probs)
                total_reward.append(running_reward)
                break
        return total_reward, mrr, loss
