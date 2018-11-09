import numpy as np
import torch

from config import config
from scripts.config_args import parse_args
from common.dataset.lc_quad import LC_QuAD
from common.model.runner import Runner

np.random.seed(6)
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    args = parse_args()
    lc_quad = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      args.remove_entity, args.remove_stop_words)
    runner = Runner(lc_quad, args)

    if args.mode == 'test':
        runner.load_checkpoint()
    else:
        runner.train(lc_quad, args)
    runner.test(lc_quad, args)