import numpy as np
import torch
import torch.nn as nn
from common.word_vectorizer.oneHotEncoder import OneHotEncoder
from tqdm import tqdm


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)
        self.activation2 = nn.Softmax()

    def forward(self, input):
        output_layer1 = self.activation1(self.layer1(input))
        output_layer2 = self.activation2(self.layer2(output_layer1))
        return output_layer2


class Agent:
    def __init__(self, number_of_relations, gamma, policy_network, policy_optimizer):
        self.gamma = gamma
        self.actions = range(number_of_relations + 1)
        self.policy_network = policy_network
        self.policy_optimizer = policy_optimizer

    def select_action(self, state):
        action_dist = self.policy_network(torch.FloatTensor(state))
        action_dist_np = np.array(action_dist.data)
        action = np.random.choice(action_dist_np, p=action_dist_np)
        action = np.argmax(action_dist_np == action)
        return action_dist, action

    def optimize(self, episode_history):
        episode_history = np.array(episode_history)
        episode_history[:, 2] = self.discount_rewards(episode_history[:, 2])

        self.policy_network.zero_grad()

        for item in episode_history:
            state, action, reward, _ = item
            action_dist, _ = self.select_action(torch.Tensor(state))
            loss = -torch.log(action_dist[action]) * reward
            loss.backward()

        self.policy_optimizer.step()

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add

        # reward_mean = np.mean(discounted_r)
        # reward_std = np.std(discounted_r)
        # for i in range(len(discounted_r)):
        #     discounted_r[i] = (discounted_r[i] - reward_mean) / reward_std

        return discounted_r


class Environment:
    def __init__(self, word_vectorizer):
        self.state = []
        self.word_vectorizer = word_vectorizer

    def init(self, input_seq):
        self.input_seq = input_seq
        self.seq_counter = -1
        self.state = self.init_state(self.next_token())
        self.action_seq = []

    def next_token(self):
        self.seq_counter += 1
        return self.word_vectorizer.decode(self.input_seq[self.seq_counter - 1]).reshape(-1)

    def is_done(self):
        return self.seq_counter == len(self.input_seq)

    def init_state(self, word):
        return [0] + ([0] * self.word_vectorizer.word_size) + list(word)

    def update_state(self, action, new_token):
        return [action] + self.state[self.word_vectorizer.word_size + 1:] + list(new_token)

    def step(self, action):
        reward = 0
        self.state = self.update_state(action, self.next_token())
        is_done = self.is_done()
        self.action_seq.append(action)

        if is_done:
            if self.action_seq == [0, 0, 0, 1, 0, 0]:
                reward = 1
            else:
                reward = -1

        return self.state, reward, is_done


class Runner:
    def __init__(self, environment, agent):
        self.environment = environment
        self.agent = agent

    def step(self, input):
        episode_history = []
        total_reward = []
        running_reward = 0
        self.environment.init(input)
        state = self.environment.state
        while True:
            _, action = self.agent.select_action(state)
            new_state, reward, done = self.environment.step(action)
            running_reward += reward
            episode_history.append([state, action, reward, new_state])
            state = new_state

            if done:
                self.agent.optimize(episode_history)
                total_reward.append(running_reward)
                break
        return total_reward


if __name__ == '__main__':
    corpus = ['who is the wife of Obama?']
    lr = 0.01
    word_vectorizer = OneHotEncoder(corpus)

    policy_network = Policy(input_size=2 * word_vectorizer.word_size + 1, output_size=2, hidden_size=8)
    agent = Agent(number_of_relations=2, gamma=0.99,
                  policy_network=policy_network, policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=lr))
    env = Environment(word_vectorizer)
    runner = Runner(environment=env, agent=agent)
    total_reward = []
    for i in tqdm(range(1000)):
        total_reward.append(runner.step(corpus[0].split()))

        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
