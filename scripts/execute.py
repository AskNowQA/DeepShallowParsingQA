import numpy as np
import torch
import argparse

from config import config
from common.dataset.lc_quad import LC_QuAD
from common.model.runner import Runner

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
    parser.add_argument('--remove_entity', dest='remove_entity', action='store_true')
    parser.add_argument('--remove_stop_words', dest='remove_stop_words', action='store_true')
    args = parser.parse_args()
    lc_quad = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      args.remove_entity, args.remove_stop_words)
    runner = Runner(lc_quad, args)

    if args.mode == 'test':
        runner.load_checkpoint()
    else:
        runner.train(lc_quad, args)
    runner.test(lc_quad, args)
