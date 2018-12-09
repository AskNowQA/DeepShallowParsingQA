import torch
import numpy as np
from common.utils import *


class Agent:
    def __init__(self, number_of_relations, gamma, policy_network, policy_optimizer):
        self.gamma = gamma
        self.actions = range(number_of_relations + 1)
        self.policy_network = policy_network
        self.policy_optimizer = policy_optimizer
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.policy_network.cuda()

    @profile
    def select_action(self, state, e, train):
        if self.cuda:
            state = state.cuda()
        action_dist = self.policy_network(state)
        m = torch.distributions.Categorical(action_dist)
        if train:
            if np.random.rand(1) < e:
                action = torch.multinomial(torch.zeros(len(action_dist)) + 0.5, self.policy_network.output_size)[0]
            else:
                action = m.sample()
            if self.cuda:
                action = action.cuda()
        else:
            action = torch.argmax(action_dist)
        return action, m.log_prob(action), action_dist.data.numpy().tolist()

    @profile
    def backward(self, rewards, action_log_probs):
        label_target = False
        if not label_target:
            rewards = self.discount_rewards(rewards)
            action_log_probs = torch.stack(action_log_probs)
            if self.cuda:
                rewards = rewards.cuda()
                action_log_probs = action_log_probs.cuda()
            loss = -torch.dot(rewards, action_log_probs)
            loss.backward()
        loss_value = float(loss)
        del loss
        return loss_value

    @profile
    def discount_rewards(self, r):
        discounted_r = torch.zeros((len(r)))
        running_add = 0
        for t in reversed(range(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add

        # reward_mean = np.mean(discounted_r)
        # reward_std = np.std(discounted_r)
        # for i in range(len(discounted_r)):
        #     discounted_r[i] = (discounted_r[i] - reward_mean) / reward_std

        return discounted_r
