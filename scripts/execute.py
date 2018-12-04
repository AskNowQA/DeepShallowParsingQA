import numpy as np
import torch
import logging
import time

from config import config
from scripts.config_args import parse_args
from common.dataset.lc_quad import LC_QuAD
from common.model.runner import Runner

np.random.seed(6)
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    lc_quad = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      False, args.remove_stop_words)
    runner = Runner(lc_quad, args)

    if args.mode == 'test':
        logger.setLevel(logging.DEBUG)
        runner.load_checkpoint()
    else:
        runner.train(lc_quad, args)
    runner.test(lc_quad, args)
    finish = time.time()
    print('total runtime:', finish - start)
