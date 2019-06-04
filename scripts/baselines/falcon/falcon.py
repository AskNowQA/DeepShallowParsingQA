from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_7_ml import Qald_7_ml
from config import config
import os
import requests
import ujson as json
from tqdm import tqdm

dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                  False, False)
dataset = dataset.test_set


# dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
#                           False, False)
# dataset = dataset.train_set

def fetch(cache_path, endpoint):
    cache = {}
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache = json.load(f)
    for idx, qarow in tqdm(enumerate(dataset)):
        question = qarow.question  # .lower()
        if question not in cache:
            try:
                payload = {'text': question}
                r = requests.post(endpoint, json=payload, timeout=120)
                results = r.json() if r.status_code == 200 else []
                cache[question] = results
                if cache_path is not None:
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
            except Exception as err:
                print('err')


def check(falcon_results, dtype='entity', k=1):
    mrrs = []
    for idx, qarow in enumerate(dataset):
        question = qarow.question  # .lower()
        if question in falcon_results:
            falcon_result = falcon_results[question]
            if dtype == 'entity':
                target_uris = qarow.sparql.entities
                if 'entities' in falcon_result:
                    falcon_result = falcon_result['entities']
                else:
                    falcon_result = []
            else:
                target_uris = qarow.sparql.relations
                if 'relations' in falcon_result:
                    falcon_result = falcon_result['relations']
                else:
                    falcon_result = []
            falcon_result = [[item[0]] for item in falcon_result]
            target_raw_uris = [target_uri.raw_uri for target_uri in target_uris]
            not_found_target_uri = list(target_raw_uris)
            change_target_uris = []

            output2 = []
            for target_uri in not_found_target_uri:
                found = False
                for candidates_idx, candidates in enumerate(falcon_result):
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
            max_len = max(len(target_uris), len(falcon_result))
            mrr = 0
            if k >= 0 and max_len > 0:
                mrr = sum(map(lambda x: 1.0 / (x + 1), rank)) / max_len
            mrrs.append(mrr)
    print(sum(mrrs) / len(mrrs))


if __name__ == '__main__':
    cache_path = './falcon_bl.cache'
    # cache_path = './falcon_bl_qald7.cache'
    # fetch(cache_path, 'https://labs.tib.eu/falcon/api?mode=long')

    cache = {}
    if cache_path is not None:
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache = json.load(f)

    for i in range(1, 2):
        check(cache, 'entity', k=i)
    for i in range(1, 2):
        check(cache, 'relation', k=i)
