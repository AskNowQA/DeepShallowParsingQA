import requests
import urllib
import ujson as json
import numpy as np
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.lc_quad import LC_QuAD
from config import config
from tqdm import tqdm

endpoint = 'https://api.dbpedia-spotlight.org/en/annotate'
headers = {
    'accept': 'application/json',
}

# cache_path = './dbps_lcquad.cache'
# dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
#                   False, False)
# cache_path = './dbps_q7.cache'
# dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
#                     False, False)
cache_path = './dbps_q6.cache'
dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
                    False, False)


def fetch(question):
    params = (
        ('text', question),
    )
    r = requests.get(endpoint, headers=headers, params=params)
    if r.status_code == 200:
        output = r.json()
        return output
    else:
        return []


def get_dbpedia_url(text):
    return "http://dbpedia.org/resource/" + text.replace(' ', '_')


def extract_dbpedia_categories(json_data):
    if 'Resources' in json_data and len(json_data['Resources']) > 0:
        return {item['@surfaceForm']: item['@URI'] for item in data[qarow.question]['Resources']}
    else:
        return {}


data = {}
for qarow in tqdm(dataset.test_set):
    data[qarow.question] = fetch(qarow.question)

with open(cache_path, 'w') as f:
    json.dump(data, f)

with open(cache_path, 'r') as f:
    data = json.load(f)

stats = []
for qarow in dataset.test_set:
    print()
    print(qarow.question)
    print(qarow.sparql.raw_sparql)
    gold_entities = [ent.raw_uri for ent in qarow.sparql.entities]
    if len(gold_entities) == 0:
        continue
    entities = extract_dbpedia_categories(data[qarow.question])
    print(gold_entities)
    print(entities)
    entities = list(entities.values())
    found_entities = [ent for ent in gold_entities if ent in entities]
    print(found_entities)
    found_entities = len(found_entities)
    f1 = found_entities / max(len(gold_entities), len(entities))
    print(f1)
    stats.append(f1)

print(np.mean(stats))
