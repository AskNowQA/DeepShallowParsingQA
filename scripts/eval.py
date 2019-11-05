import numpy as np
import torch
import logging
import time
import os
import ujson as json

from config import config
from scripts.config_args import parse_args
from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_7_ml import Qald_7_ml
from common.model.runner import Runner

np.random.seed(6)
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    dataset = None
    if args.dataset == 'lcquad':
        dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                          False, args.remove_stop_words)
    elif args.dataset == 'qald_7_ml':
        dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
                            False, False)

    try:
        with open('eval-{}.json'.format(args.dataset), 'rt') as json_file:
            eval_results = json.load(json_file)
    except:
        eval_results = {}

    if args.mode == 'test':
        for file_name in os.listdir(config['chk_path']):
            if file_name.startswith(args.dataset) and 'bilstm' in file_name:
                try:
                    l = len(args.dataset)
                    args.policy = file_name[l + 1:-6]
                    args.b = int(file_name[-4:-3])
                    args.k = 1
                    args.checkpoint = file_name
                    runner = Runner(dataset, args)
                    runner.load_checkpoint(os.path.join(config['chk_path'], args.checkpoint))
                    print(args)
                    results = runner.test(dataset, args, use_elastic=True)  # use_EARL=True)
                    eval_results[file_name] = results
                    finish = time.time()
                    print('total runtime:', finish - start)
                except Exception as e:
                    print(e)
                    print(file_name)
                    eval_results[file_name] = [0,0]

        with open('eval-{}.json'.format(args.dataset), 'wt') as json_file:
            json.dump(eval_results, json_file)
