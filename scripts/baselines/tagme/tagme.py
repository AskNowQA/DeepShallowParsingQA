import requests
import urllib
import ujson as json
import numpy as np
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.lc_quad import LC_QuAD
from config import config


endpoint = 'https://tagme.d4science.org/tagme/tag'
payload = {'lang': 'en',
           'gcube-token': '6904c806-096d-4c76-b7f0-b35a52488fe2-843339462',
           'include_categories': True,
           'text': ''}
# cache_path = './tagme_lcquad.cache'
# dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
#                   False, False)
cache_path = './tagme_q7.cache'
dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
                    False, False)


def fetch(question):
    payload['text'] = question
    query_string = urllib.parse.urlencode(payload)
    url = endpoint + '?' + query_string
    r = requests.get(url)
    if r.status_code == 200:
        output = r.json()
        return output
    else:
        return []


def get_dbpedia_url(text):
    return "http://dbpedia.org/resource/" + text.replace(' ', '_')


def extract_dbpedia_categories(json_data):
    if 'annotations' in json_data and len(json_data['annotations']) > 0 and 'title' in \
            json_data['annotations'][0]:
        return {item['spot']: get_dbpedia_url(item['title']) for item in json_data['annotations']}


# data = {}
# for qarow in dataset.test_set:
#     data[qarow.question] = fetch(qarow.question)
#
# with open(cache_path, 'w') as f:
#     json.dump(data, f)

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
