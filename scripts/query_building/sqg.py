from common.utils import Utils
import config
import itertools


class SQG:
    def __init__(self, endpoint=config.config['SQG']['endpoint'], timeout=config.config['SQG']['timeout']):
        self.endpoint = endpoint
        self.timeout = timeout

    def build_query(self, question, entities=[], relations=[], boolean_query=True, count_query=True):
        input = {'question': question,
                 'entities': entities,
                 'relations': relations,
                 'timeout': self.timeout,
                 'use_cache': config.config['SQG']['use_sqg_cache'],
                 'force_list': True}
        result_list = None
        result_bool = None
        result_count = None

        if boolean_query:
            input['force_list'] = False
            input['force_bool'] = True
            result_bool = Utils.call_web_api(self.endpoint, input)
        else:
            result_list = Utils.call_web_api(self.endpoint, input)

        if result_list is not None and count_query:
            input['force_list'] = False
            input['force_bool'] = False
            input['force_count'] = True
            result_count = Utils.call_web_api(self.endpoint, input)
        result = {'queries': []}
        for queries in itertools.chain([result_list, result_bool, result_count]):
            if queries is None:
                continue
            for query in queries['queries']:
                query['type'] = queries['type']
                query['type_confidence'] = queries['type_confidence']
            result['queries'].extend(queries['queries'])

        return result


if __name__ == '__main__':
    print('SQG')
    sqg = SQG('http://127.0.0.1:5011/qg/api/v1.0/query', 120)
    ents = [{'surface': [3, 15], 'uris': [{'confidence': 1.0, 'uri': 'http://dbpedia.org/resource/Bill_Finger'}]}]
    rels = [{'surface': [30, 5], 'uris': [{'confidence': 1.0, 'uri': 'http://dbpedia.org/ontology/creator'}]}]
    print(sqg.build_query('stuff Bill Finger made?', ents, rels, False, False))
