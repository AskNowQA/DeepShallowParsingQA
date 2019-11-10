from common.linkers.candidate_generator.earlCG import EARLCG
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.lc_quad import LC_QuAD
from config import config
import json

# dataset_name = 'lcquad'
# dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
#                   False, False)
# dataset = dataset.test_set
# earlCG = EARLCG(config['EARL']['endpoint'], './earl_bl.cach')


# dataset_name='qald_7_ml'
# dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
#                           False, False)
# earlCG = EARLCG(config['EARL']['endpoint'], './earl_bl_qald7.cach')
# dataset = dataset.test_set

dataset_name = 'qald_6_ml'
dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
                    False, False)
earlCG = EARLCG(config['EARL']['endpoint'], './earl_bl_qald6.cach')
dataset = dataset.test_set


def fetch():
    for idx, qarow in enumerate(dataset):
        question = qarow.question
        earl_result = earlCG.fetch(question)
        if earl_result is not None:
            print(idx)


def check(earl_results, dtype='entity', k=1):
    mrrs = []
    for idx, qarow in enumerate(dataset):
        question = qarow.question
        earl_result = earl_results[question]
        target_uris = qarow.sparql.entities + qarow.sparql.relations
        target_raw_uris = [target_uri.raw_uri for target_uri in target_uris]
        not_found_target_uri = list(target_raw_uris)
        change_target_uris = []

        output2 = []
        for target_uri in not_found_target_uri:
            found = False
            for candidates_idx, candidates in enumerate(earl_result.values()):
                type = candidates[0]
                if type != dtype:
                    continue
                candidates = candidates[1]
                candidates = [item[0] for item in candidates]
                number_of_candidates = len(candidates)
                if target_uri in candidates:
                    idx = candidates.index(target_uri)
                    while idx >= 1 and candidates[idx].lower() == candidates[idx - 1].lower():
                        idx -= 1
                    score = 1 - idx / number_of_candidates
                    output2.append([target_uri, candidates_idx, score, idx, ''])
                    found = True
            # if not found and target_uri not in change_target_uris:
            #     if '/property/' in target_uri:
            #         new_uri = target_uri.replace('/property/', '/ontology/')
            #         not_found_target_uri.append(new_uri)
            #         change_target_uris.append(new_uri)
            #     elif '/ontology/' in target_uri:
            #         new_uri = target_uri.replace('/ontology/', '/property/')
            #         not_found_target_uri.append(new_uri)
            #         change_target_uris.append(new_uri)
        output2.sort(key=lambda x: x[2], reverse=True)
        used_uris, used_candidates, used_surfaces, rank, found_uris = [], [], [], [], []
        for item in output2:
            if item[0] in used_uris or item[1] in used_candidates:
                pass
            else:
                used_uris.append(item[0])
                used_candidates.append(item[1])

                if item[3] <= k:
                    rank.append(item[3])
                    found_uris.append(item[0])
        max_len = max(len(qarow.sparql.entities if dtype == 'entity' else qarow.sparql.relations), len(
            [item for item in earl_result.values() if item[0] == dtype]))
        mrr = 0
        if k >= 0 and max_len > 0:
            mrr = sum(map(lambda x: 1.0 / (x + 1), rank)) / max_len
        mrrs.append(mrr)
    result = sum(mrrs) / len(mrrs)
    print(result)
    return result


if __name__ == '__main__':
    # fetch()
    eval_results = {}
    for k in range(0, 11):
        file_name = '{}-{}'.format(dataset_name, k)
        ent = check(earlCG.cache, 'entity', k=k)
        rel = check(earlCG.cache, 'relation', k=k)
        eval_results[file_name] = [ent, rel]
    with open('earl-mrr-{}.json'.format(dataset_name), 'wt') as json_file:
        json.dump(eval_results, json_file)
