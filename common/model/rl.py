import numpy as np
import torch
import torch.nn as nn
from common.word_vectorizer.glove import Glove
from common.word_vectorizer.oneHotEncoder import OneHotEncoder
from tqdm import tqdm
from config import config
import ujson as json
import random

np.random.seed(6)
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True

global_target = []
global_loss = 0


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()

        self.activation0 = nn.GLU()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        self.activation1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.layer2 = nn.Linear(int(hidden_size), output_size, bias=False)
        self.activation2 = nn.Softmax()

        # self.layer1.reset_parameters()
        # self.layer2.reset_parameters()

    def forward(self, input):
        output_layer1 = self.activation1(self.layer1(input))
        output_layer1 = self.dropout(output_layer1)
        output_layer2 = self.activation2(self.layer2(output_layer1))
        return output_layer2


class Agent:
    def __init__(self, number_of_relations, gamma, policy_network, policy_optimizer):
        self.gamma = gamma
        self.actions = range(number_of_relations + 1)
        self.policy_network = policy_network
        self.policy_optimizer = policy_optimizer

    def select_action(self, state, e):
        action_dist = self.policy_network(torch.FloatTensor(state))
        action_dist_np = action_dist
        m = torch.distributions.Categorical(action_dist_np)
        if np.random.rand(1) < e:
            action = torch.multinomial(torch.zeros(len(action_dist)) + 0.5, 1)[0]
        else:
            action = m.sample()
        return action_dist, action, m.log_prob(action)

    def optimize(self, episode_history):
        label_target = False

        self.policy_network.zero_grad()

        if not label_target:
            correct = episode_history[:, 2][-1] > 0
            action_prob = torch.stack(list(episode_history[:, 1])).double()
            rewards = torch.from_numpy(self.discount_rewards(episode_history[:, 2]))
            action_prob.log()
            loss = -torch.dot(rewards, torch.stack(list(episode_history[:, 6])).double())  # .clamp(min=1e-6)
            global_loss = loss
            # print(global_loss)
            if correct:
                loss *= 1
            loss.backward()
        else:
            loss = nn.CrossEntropyLoss()
            labels = torch.stack(list(episode_history[:, 4])).double().view(-1, 2)
            targets = torch.LongTensor(global_target).view(-1)
            output = loss(labels, targets)
            output.backward()

        self.policy_optimizer.step()
        if torch.isnan(torch.sum(self.policy_network.layer1.weight)):
            print('noo')

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r, dtype=float)
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
    def __init__(self):
        self.state = []
        self.target = []

    def init(self, input_seq):
        self.input_seq = input_seq
        self.seq_counter = -1
        self.state = self.init_state(self.next_token())
        self.action_seq = []

    def next_token(self):
        self.seq_counter += 1
        return self.input_seq[self.seq_counter - 1].reshape(-1)

    def is_done(self):
        return self.seq_counter == len(self.input_seq)  # or np.sum(self.action_seq) > 2

    def init_state(self, word):
        return torch.cat((torch.FloatTensor([1]), word))
        # return torch.cat((torch.FloatTensor([1]), torch.zeros(word.size(0)), word))

    def update_state(self, action, new_token):
        return torch.cat((torch.FloatTensor([action]), new_token))
        # return torch.cat((torch.FloatTensor([action]), torch.zeros(new_token.size(0)), new_token))

    def step(self, action):
        reward = 0
        self.state = self.update_state(action, self.next_token())
        self.action_seq.append(action)

        is_done = self.is_done()

        if is_done:
            if self.action_seq == self.target:
                reward = 10
            else:
                reward = -1

        return self.state, reward, is_done

    def set_target(self, target):
        self.target = target


class Runner:
    def __init__(self, environment, agent, word_vectorizer):
        self.environment = environment
        self.agent = agent
        self.word_vectorizer = word_vectorizer
        self.unique_action_seq = []

    def step(self, input, e):
        episode_history = []
        total_reward = []
        running_reward = 0
        self.environment.init(self.word_vectorizer.decode(input))
        state = self.environment.state
        while True:
            action_dist, action, m = self.agent.select_action(state, e)
            new_state, reward, done = self.environment.step(action)
            running_reward += reward
            episode_history.append([state, action_dist[action], reward, new_state, action_dist, action, m])
            state = new_state

            if done:
                episode_history = np.array(episode_history)
                action_seq = list(episode_history[:, 1])
                f = False
                for item in self.unique_action_seq:
                    if action_seq == item:
                        f = True
                if not f:
                    self.unique_action_seq.append(action_seq)
                self.agent.optimize(episode_history)
                total_reward.append(running_reward)
                break
        return total_reward


if __name__ == '__main__':
    # corpus = [
    #     ['who is the wife of Obama', [0, 0, 0, 1, 0, 0]],
    #     ['who is Obama wife', [0, 0, 0, 1]]
    # ]

    corpus = [
        # ['how many direct', [0, 0, 1]],
        # ['which city s founder', [0, 1, 0, 1]],
        ['How many movies did Stanley Kubrick direct', [0, 0, 0, 0, 0, 0, 1]],
        ['Which city s foundeer is John Forbes', [0, 1, 0, 1, 0, 0, 0]],
        ['What is the river whose mouth is in deadsea', [0, 0, 0, 1, 0, 1, 0, 0, 0]]
    ]

    # with open(config['lc_quad']['tiny'], 'r') as file_hanlder:
    #     raw_dataset = json.load(file_hanlder)
    #     corpus = [[item['corrected_question'], item['annotation']] for item in raw_dataset]

    lr = 0.001
    # word_vectorizer = OneHotEncoder([item[0] for item in corpus])
    word_vectorizer = Glove([item[0] for item in corpus], config['glove_path'])

    policy_network = Policy(input_size=word_vectorizer.word_size + 1,
                            hidden_size=int(word_vectorizer.word_size * 2),
                            output_size=2)
    agent = Agent(number_of_relations=2, gamma=0.9,
                  policy_network=policy_network, policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=lr))
    env = Environment()
    runner = Runner(environment=env, agent=agent, word_vectorizer=word_vectorizer)
    total_reward = []
    last_idx = 0
    e = 1
    for i in tqdm(range(4000)):
        for doc in corpus:
            # for idx in range(10):
            env.set_target(doc[1])
            global_target = doc[1]
            total_reward.append(runner.step(doc[0], e))

        if i % 100 == 0:
            print(np.mean(total_reward[last_idx:]),
                  np.sum(np.array(total_reward[last_idx:]) > 0) / len(total_reward[last_idx:]), e)
            last_idx = len(total_reward)

        e = 0.5 / (i / 100 + 1)
        # if e < 0.1:
        #     e = 0.1
