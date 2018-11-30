import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Shallow Parsing for QA')
    parser.add_argument('--mode', default='train', help='mode: `train` or `test`')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.9, type=float, help='gamma')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--k', default=10, type=int, help='top-k candidate')
    parser.add_argument('--e', default=0.001, type=float, help='epsilon-greedy value')
    parser.add_argument('--batchsize', default=100, type=int, help='batchsize for optimizer updates')
    parser.add_argument('--positive_reward', default=1, type=float, help='positive reward')
    parser.add_argument('--negative_reward', default=-0.5, type=float, help='negative reward')
    # parser.add_argument('--remove_entity', dest='remove_entity', action='store_true')
    parser.add_argument('--remove_stop_words', dest='remove_stop_words', action='store_true')
    args = parser.parse_args()
    return args
