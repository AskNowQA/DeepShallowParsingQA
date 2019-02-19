from tqdm import tqdm
import pickle as pk
from config import config
from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_7_ml import Qald_7_ml
from common.kb.dbpedia import DBpedia
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate one hop ')
    parser.add_argument('--dataset', default='lc_quad', help='`lc_quad` or `qald_7_ml`')
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
    kb = DBpedia()

    one_hop = {}
    for qa_row in tqdm(dataset.test_set):
        for entity in qa_row.sparql.entities:
            if entity.raw_uri not in one_hop:
                relations = []
                relations = kb.one_hop_relations(entity.raw_uri)
                relations = [[item, item[item.rindex('/') + 1:]] for item in relations]
                if relations is not None:
                    one_hop[entity.raw_uri] = relations
        with open(file_name, 'wb') as f:
            pk.dump(one_hop, f)

    with open(file_name, 'rb') as f:
        one_hop = pk.load(f)
    print(len(one_hop))
