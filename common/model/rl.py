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
    lc_quad = LC_QuAD(config['lc_quad']['tiny'], config['lc_quad']['vocab'])
    print(lc_quad.corpus)

    lr = 0.0001
    word_vectorizer = Glove(lc_quad, config['glove_path'], config['lc_quad']['emb'])

    policy_network = Policy(vocab_size=lc_quad.vocab.size(),
                            emb_size=word_vectorizer.word_size,
                            input_size=word_vectorizer.word_size * 2 + 1,
                            hidden_size=int(word_vectorizer.word_size * 4),
                            output_size=2,
                            dropout_ratio=0.5)

    policy_network.emb.weight.data.copy_(word_vectorizer.emb)

    agent = Agent(number_of_relations=2, gamma=0.9,
                  policy_network=policy_network, policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=lr))
    linker = OrderedLinker(sorter=StringSimilaritySorter(),
                           rel2id_path=config['lc_quad']['rel2id'],
                           core_chains_path=config['lc_quad']['core_chains'],
                           dataset=lc_quad)
    env = Environment(linker=linker, positive_reward=1, negative_reward=-0.5)
    runner = Runner(environment=env, agent=agent)

    ###### Test
    print('Train')
    total_reward = []
    last_idx = 0
    e = 0.001
    for i in tqdm(range(100)):
        for idx, qarow in enumerate(lc_quad.dataset):
            total_reward.append(runner.step(lc_quad.coded_corpus[idx], qarow, e))

        if i % 10 == 0:
            print(np.mean(total_reward[last_idx:]),
                  np.sum(np.array(total_reward[last_idx:]) > 0) / len(total_reward[last_idx:]), e)
            last_idx = len(total_reward)

    ###### Test
    print('test')
    total_reward = []
    last_idx = 0
    for idx, qarow in enumerate(lc_quad.dataset):
        total_reward.append(runner.step(lc_quad.coded_corpus[idx], qarow, e, False))

    print(np.mean(total_reward[last_idx:]),
          np.sum(np.array(total_reward[last_idx:]) > 0) / len(total_reward[last_idx:]), e)
    last_idx = len(total_reward)

        # e = 1 / (i / 100 + 1)
        # if e < 0.1:
        #     e = 0.1
