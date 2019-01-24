from tqdm import tqdm
import pickle as pk
from config import config
from common.dataset.lc_quad import LC_QuAD
from common.kb.dbpedia import DBpedia

file_name = config['lc_quad']['entity_one_hop']
lc_quad = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'], False, True)

kb = DBpedia()

one_hop = {}
for qa_row in tqdm(lc_quad.test_set):
    for entity in qa_row.sparql.entities:
        if entity.raw_uri not in one_hop:
            relations = []
            relations = kb.one_hop_relations(entity.raw_uri)
            if relations is not None:
                one_hop[entity.raw_uri] = relations
    with open(file_name, 'wb') as f:
        pk.dump(one_hop, f)

with open(file_name, 'rb') as f:
    one_hop = pk.load(f)
print(len(one_hop))
