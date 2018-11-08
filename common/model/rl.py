import numpy as np
import torch
from tqdm import tqdm
import argparse

from config import config
from common.word_vectorizer.glove import Glove
from common.dataset.lc_quad import LC_QuAD
from common.model.agent import Agent
from common.model.policy import Policy
from common.model.environment import Environment
from common.model.runner import Runner
from common.linkers.orderedLinker import OrderedLinker
from common.linkers.stringSimilaritySorter import StringSimilaritySorter
from common.linkers.embeddingSimilaritySorter import EmbeddingSimilaritySorter

np.random.seed(6)
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shallow Parsing for QA')
    parser.add_argument('--mode', default='train', help='mode: `train` or `test`')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.9, type=float, help='gamma')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--k', default=10, type=int, help='top-k candidate')
    parser.add_argument('--e', default=0.001, type=float, help='epsilon-greedy value')
    parser.add_argument('--sim', default='str', help='similarity (default: str) str or emb')
    args = parser.parse_args()

    lc_quad = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'])

    word_vectorizer = Glove(lc_quad, config['glove_path'], config['lc_quad']['emb'])

    policy_network = Policy(vocab_size=lc_quad.vocab.size(),
                            emb_size=word_vectorizer.word_size,
                            input_size=word_vectorizer.word_size * 2 + 1,
                            hidden_size=int(word_vectorizer.word_size * 4),
                            output_size=2,
                            dropout_ratio=args.dropout)

    policy_network.emb.weight.data.copy_(word_vectorizer.emb)
    agent = Agent(number_of_relations=2,
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
    env = Environment(linker=linker, positive_reward=1, negative_reward=-0.5)
    runner = Runner(environment=env, agent=agent)

    checkpoint_filename = 'checkpoint.pyt'
    if args.mode == 'test':
        checkpoint = torch.load(checkpoint_filename)
        policy_network.load_state_dict(checkpoint['model'])
    else:
        ###### Train
        print('Train')
        total_reward = []
        total_rmm = []
        last_idx = 0

        for epoch in tqdm(range(args.epochs)):
            for idx, qarow in enumerate(lc_quad.train_set):
                reward, mrr = runner.step(lc_quad.coded_train_corpus[idx], qarow, e=args.e, k=args.k)
                total_reward.append(reward)
                total_rmm.append(mrr)

            if epoch > 0 and epoch % 10 == 0:
                print(np.mean(total_reward[last_idx:]), np.mean(total_rmm[last_idx:]))
                last_idx = len(total_reward)

                checkpoint = {'model': policy_network.state_dict(), 'epoch': epoch}
                torch.save(checkpoint, checkpoint_filename)

        print(np.mean(total_reward[last_idx:]), np.mean(total_rmm[last_idx:]))

    ###### Test
    print('Test')
    total_rmm = []
    last_idx = 0
    for idx, qarow in enumerate(lc_quad.test_set):
        reward, mrr = runner.step(lc_quad.coded_test_corpus[idx], qarow, e=args.e, train=False, k=args.k)
        total_rmm.append(mrr)

    print(np.mean(total_rmm))
    # e = 1 / (i / 100 + 1)
    # if e < 0.1:
    #     e = 0.1
