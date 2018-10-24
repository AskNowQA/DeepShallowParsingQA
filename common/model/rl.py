import numpy as np
import torch
from tqdm import tqdm

from config import config
from common.word_vectorizer.glove import Glove
from common.dataset.lc_quad import LC_QuAD
from common.model.agent import Agent
from common.model.policy import Policy
from common.model.environment import Environment
from common.model.runner import Runner
from common.linkers.orderedLinker import OrderedLinker
from common.linkers.stringSimilaritySorter import StringSimilaritySorter

np.random.seed(6)
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    dataset = LC_QuAD(config['lc_quad']['tiny'])

    lr = 0.0001
    word_vectorizer = Glove([item[0] for item in dataset.corpus], config['glove_path'])

    policy_network = Policy(input_size=word_vectorizer.word_size + 1,
                            hidden_size=int(word_vectorizer.word_size * 2),
                            output_size=2,
                            dropout_ratio=0.5)
    agent = Agent(number_of_relations=2, gamma=0.9,
                  policy_network=policy_network, policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=lr))
    linker = OrderedLinker(sorter=StringSimilaritySorter(),
                           rel2id_path=config['lc_quad']['rel2id'],
                           core_chains_path=config['lc_quad']['core_chains'],
                           dataset=dataset)
    env = Environment(linker=linker, positive_reward=1, negetive_reward=-0.5)
    runner = Runner(environment=env, agent=agent, word_vectorizer=word_vectorizer)
    total_reward = []
    last_idx = 0
    e = 0.001
    for i in tqdm(range(4000)):
        for doc in dataset.corpus:
            env.set_target(doc[2])
            total_reward.append(runner.step(doc[1], doc[0], e))

        if i % 100 == 0:
            print(np.mean(total_reward[last_idx:]),
                  np.sum(np.array(total_reward[last_idx:]) > 0) / len(total_reward[last_idx:]), e)
            last_idx = len(total_reward)

        # e = 1 / (i / 100 + 1)
        # if e < 0.1:
        #     e = 0.1
