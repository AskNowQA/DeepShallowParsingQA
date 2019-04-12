import requests
import os
import ujson as json


class EARLCG:
    def __init__(self, endpoint, cache_path):
        self.endpoint = endpoint
        self.cache_path = cache_path
        self.cache = {}
        if cache_path is not None:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    self.cache = json.load(f)

    def fetch(self, question):
        if question in self.cache:
            return self.cache[question]
        payload = {
            'nlquery': question,
            'pagerankflag': False,
        }
        try:
            r = requests.post(self.endpoint, json=payload)
            results = r.json() if r.status_code == 200 else []
            results = {chunk['chunk']: [results['ertypes'][idx],list(
                map(lambda x: [x[1], x[1][x[1].rindex('/') + 1:] if '/' in x[1] else x[1]],
                    results['rerankedlists'][str(idx)]))]
                for idx, chunk in enumerate(results['chunktext'])}
            if question not in self.cache:
                self.cache[question] = results
            else:
                for key, value in results.items():
                    self.cache[question][key] = value

            if self.cache_path is not None:
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f)
        except Exception as err:
            print('err')
        if question in self.cache:
            return self.cache[question]
        else:
            return None

    def __fetch(self, surfaces, extra_surfaces, surface, question):
        surfaces_joined = [{'chunk': ' '.join(item), 'class': 'entity'} for item in surfaces]
        extra_surfaces_joined = [{'chunk': ' '.join(item), 'class': 'relation'} for item in extra_surfaces]
        payload = {
            'erpredictions': surfaces_joined + extra_surfaces_joined,
            'nlquery': question,
            'pagerankflag': False,
        }
        try:
            r = requests.post(self.endpoint, json=payload)
            results = r.json() if r.status_code == 200 else []
            results = {chunk['chunk']: list(
                map(lambda x: [x[1], x[1][x[1].rindex('/') + 1:] if '/' in x[1] else x[1]],
                    results['rerankedlists'][str(idx)]))
                for idx, chunk in enumerate(results['chunktext'])}
            if question not in self.cache:
                self.cache[question] = results
            else:
                for key, value in results.items():
                    self.cache[question][key] = value

            if self.cache_path is not None:
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f)
        except Exception as err:
            print('err')

    def generate(self, surfaces, extra_surfaces, surface, question, size=100):
        if question not in self.cache:
            self.__fetch(surfaces, extra_surfaces, surface, question)
        output = self.cache[question]
        surface = surface.lower()
        if surface in map(str.lower, self.cache[question].keys()):
            for key, value in output.items():
                if key.lower() == surface:
                    return output[key]
        else:
            self.__fetch(surfaces, extra_surfaces, surface, question)
            output = self.cache[question]
            if surface in output:
                return output[surface]
            else:
                print('ffff')

        return []

# e = EARLCG('http://131.220.9.219/iqa/processQuery')
# e.generate([],[], ',', 'who is the president of Russia?')
