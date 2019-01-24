import requests
from config import config
import urllib
import os


class DBpedia:
    def __init__(self, endpoint=config['dbpedia_kb']['endpoint']):
        self.endpoint = endpoint

    def query(self, q):
        payload = (
            ('query', q),
            ('format', 'application/json'))
        try:
            r = requests.get(self.endpoint, params=payload, timeout=60)
        except:
            return 0, None

        return r.status_code, r.json() if r.status_code == 200 else None

    def one_hop_relations(self, entity):
        query = '''SELECT distinct  ?r WHERE {{
        {{<{uri}> ?r ?x. FILTER (regex(?r, "dbpedia") && !regex(?r, "wikiPage"))}} 
        UNION
        {{?x ?r <{uri}>. FILTER (regex(?r, "dbpedia") && !regex(?r, "wikiPage"))}} 
        }}
        '''
        code, results = self.query(query.format(uri=entity))
        if code == 200:
            return [item['r']['value'] for item in results['results']['bindings'] if item['r']['type'] == 'uri']
        else:
            return None
