import re
import os
import ujson as json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common.linkers.candidate_generator.earlCG import EARLCG
from config import config
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.lc_quad import LC_QuAD

# dataset_name = 'qald6'
# dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
#                     False, False)

# dataset_name = 'qald7'
# dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
#                               False, False)

dataset_name = 'lcquad'
dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                  False, False)

font = {'family': 'normal',
        # 'weight' : 'bold',
        'size': 12}

matplotlib.rc('font', **font)


def check_linker(qapair, entities=[], relations=[]):
    if qapair is not None:
        wrong_ent = True
        wrong_rel = True
        if len(entities) == len(qapair.sparql.entities):
            wrong_ent = len(
                [uri_o for uri_o in qapair.sparql.entities if
                 uri_o.raw_uri not in [uri for item in entities for uri in item]]) > 0
        target_rels = qapair.sparql.relations
        # if '#type' in qapair.sparql.raw_sparql and len(relations) == len(target_rels) - 1 and len(
        #         relations) == 2:
        #     type_uri = \
        #         re.findall('(<[^>]*>|\?[^ ]*)',
        #                    qapair.sparql.raw_sparql[qapair.sparql.raw_sparql.index('#type'):])[0]
        #     target_rels = [uri_o for uri_o in qapair.sparql.relations if uri_o.raw_uri != type_uri]
        # else:
        if len(relations) != len(target_rels):
            return False

        wrong_rel = len(
            [uri_o for uri_o in target_rels if uri_o.raw_uri not in [uri for item in relations for uri in item]]) > 0
        return not (wrong_ent or wrong_rel)
    return True


def check(linker_results, dataset, get_fn, k):
    results = {}
    i = 0
    for idx, qarow in enumerate(dataset):
        question = qarow.question
        item_result = linker_results[question]
        entities, relations = get_fn(item_result, k)
        results[question] = check_linker(qarow, entities, relations)
        i = len(results)
    return results


def get_earl_item(item_result, k=100):
    relations = [[o[0] for o in item_result[item][1]][:k] for item in item_result if item_result[item][0] == 'relation']
    entities = [[o[0] for o in item_result[item][1]][:k] for item in item_result if item_result[item][0] == 'entity']
    return entities, relations


def get_falcon_item(item_result, k=100):
    if len(item_result) > 0:
        relations = [[item[0]][:k] for item in item_result['relations']]
        entities = [[item[0]][:k] for item in item_result['entities']]
        return entities, relations
    else:
        return [], []


def get_mdp_item(item_result, k=100):
    if len(item_result) > 0:
        relations = [[o['uri'] for o in item['uris']][:k] for item in item_result['relations']]
        entities = [[o['uri'] for o in item['uris']][:k] for item in item_result['entities']]
        return entities, relations
    else:
        return [], []


if __name__ == '__main__':

    earlCG = EARLCG(config['EARL']['endpoint'],
                    '/Users/hamid/workspace/DeepShallowParsingQA/scripts/baselines/earl/earl_bl_{}.cach'.format(
                        dataset_name))
    cache_path = '/Users/hamid/workspace/DeepShallowParsingQA/scripts/baselines/falcon/falcon_bl_{}.cache'.format(
        dataset_name)
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                falcon_cache = json.load(f)
    cache_path = '/Users/hamid/workspace/DeepShallowParsingQA/scripts/data/mdp-{}.cache'.format(dataset_name)
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                mdp_cache = json.load(f)

    cache_path = '/Users/hamid/workspace/DeepShallowParsingQA/scripts/data/mdp+earl-{}.cache'.format(dataset_name)
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                mdpearl_cache = json.load(f)

    stats = []
    xrange = range(1, 10)
    for k in xrange:
        results_earl = check(earlCG.cache, dataset.test_set, get_earl_item, k)
        results_falcon = check(falcon_cache, dataset.test_set, get_falcon_item, k)
        results_mdp = check(mdp_cache, dataset.test_set, get_mdp_item, k)
        results_mdpearl = check(mdpearl_cache, dataset.test_set, get_mdp_item, k)
        stats.append([sum(results_earl.values()),
                      sum(results_falcon.values()),
                      sum(results_mdp.values()),
                      sum(results_mdpearl.values())])
    stats = np.array(stats) / len(dataset.test_set)
    print(stats)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(xrange, stats[:, 0], marker='o', color='black', label='EARL')
    ax.plot(xrange, stats[:, 3], marker='^', color='blue', label='EARL+MDP-Parser')
    ax.plot(xrange, stats[:, 1], marker='>', color='red', label='Falcon')
    ax.plot(xrange, stats[:, 2], marker='x', color='green', label='MDP-Parser')
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1.02), borderaxespad=0.)
    plt.xlabel('k')
    plt.ylabel('Recall@k')
    fig.tight_layout()
    plt.savefig('../figs/qa.png')
    # plt.show()
