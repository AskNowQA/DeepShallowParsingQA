import argparse
import re
import torch
import os
import pickle as pk
from tqdm import tqdm
from config import config
from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.simple_dbpedia_qa import SimpleDBpediaQA
from common.kb.dbpedia import DBpedia

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate one hop ')
    parser.add_argument('--dataset', default='lc_quad', help='`lc_quad` or `qald_7_ml`')
    parser.add_argument('--max_length', default=3, type=int)
    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name == 'lc_quad':
        file_name = config['lc_quad']['entity_one_hop']
        dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'], False,
                          True)
    elif dataset_name == 'qald_7_ml':
        file_name = config['qald_7_ml']['entity_one_hop']
        dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
                            False, False)
    elif dataset_name == 'qald_6_ml':
        file_name = config['qald_6_ml']['entity_one_hop']
        dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
                            False, False)
    elif args.dataset == 'simple':
        file_name = config['SimpleDBpediaQA']['entity_one_hop']
        dataset = SimpleDBpediaQA(config['SimpleDBpediaQA']['train'], config['SimpleDBpediaQA']['test'],
                                  config['SimpleDBpediaQA']['vocab'],
                                  False, False)
    kb = DBpedia()

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            one_hop = pk.load(f)
    else:
        one_hop = {}
    for qa_row in tqdm(dataset.test_set + dataset.train_set):
        for entity in qa_row.sparql.entities:
            if entity.raw_uri not in one_hop:
                relations = []
                relations = kb.one_hop_relations(entity.raw_uri)
                if relations is not None:
                    relations = [[item, item[item.rindex('/') + 1:]] for item in relations]
                    one_hop[entity.raw_uri] = relations
    with open(file_name, 'wb') as f:
        pk.dump(one_hop, f)

    vocab = dataset.vocab
    for entity, uris in one_hop.items():
        for idx in range(len(uris)):
            uri, label = uris[idx][:2]
            label = re.sub(r"([A-Z])", r" \1", label).replace('_', ' ').replace('.', ' ')
            words = list(map(str.lower, label.split(' ')))
            coded = vocab.convertToIdx(words, '')
            coded_length = min(len(coded), args.max_length)
            if len(coded) < args.max_length:
                coded = coded + [0] * (args.max_length - len(coded))
            else:
                coded = coded[:args.max_length]
            coded = torch.LongTensor(coded)
            if len(uris[idx]) == 2:
                uris[idx].extend([coded, coded_length])
            else:
                uris[idx][2] = coded
                uris[idx][3] = coded_length
    with open(file_name, 'wb') as f:
        pk.dump(one_hop, f)

    print(len(one_hop))
