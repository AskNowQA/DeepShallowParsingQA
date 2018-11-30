from elasticsearch import Elasticsearch
from config import config
import ujson as json
from tqdm import tqdm


class Elastic:
    def __init__(self, server, entity_index_config, entities_path, create_entity_index=False):
        self.es = Elasticsearch(hosts=[server])

        if create_entity_index:
            batch_size = 100000
            delete_index = True

            index_name = 'idx'
            type_name = 'resources'
            bulk_data = []
            counter = 0
            with open(entities_path, 'r') as file_handler:
                for line in tqdm(file_handler):
                    json_object = json.loads(line)['_source']
                    if 'http://dbpedia.org/' in json_object['uri']:
                        dtype = 'uri'
                    else:
                        dtype = 'literal'
                    label = ''
                    if 'dbpediaLabel' in json_object:
                        label = json_object['dbpediaLabel']
                    elif 'wikidataLabel' in json_object:
                        label = json_object['wikidataLabel']
                    if 'dbpediaLabel' in json_object and 'wikidataLabel' in json_object:
                        print(json_object)

                    data_dict = {'key': json_object['uri'],
                                 'dtype': dtype,
                                 'label': label,
                                 'edge_count': json_object['edgecount']}
                    op_dict = {"index": {"_index": index_name, "_type": type_name, "_id": json_object['uri']}}
                    bulk_data.append(op_dict)
                    bulk_data.append(data_dict)
                    if counter > 0 and counter % batch_size == 0:
                        res = self.__bulk_indexing(index_name='idx',
                                                   delete_index=delete_index,
                                                   index_config=entity_index_config,
                                                   bulk_data=bulk_data)
                        bulk_data = []
                        delete_index = False
                    counter += 1
                if len(bulk_data) > 0:
                    res = self.__bulk_indexing(index_name='idx',
                                               delete_index=delete_index,
                                               index_config=entity_index_config,
                                               bulk_data=bulk_data)

    def __bulk_indexing(self, index_name, delete_index, index_config, bulk_data):
        if delete_index:
            if self.es.indices.exists(index_name):
                print("deleting '{}' index...".format(index_name))
                res = self.es.indices.delete(index=index_name)
                print(" response: '{}'".format(res))

            print("creating '{}' index...".format(index_name))
            res = self.es.indices.create(index=index_name, body=index_config)
            print(" response: '{}'".format(res))

        print(len(bulk_data))
        print("bulk indexing...")
        res = self.es.bulk(index=index_name, body=bulk_data, refresh=True)
        print("bulk indexing done")
        return res

    def search_ngram(self, text, index, constraint=None):
        if constraint is None:
            results = self.es.search(index=index, doc_type='resources', body={
                'query': {'match': {'label': text, }}
            })
        else:
            results = self.es.search(index=index, doc_type='resources', body={
                'query': {'bool': {'must': [{'match': {'label': text}}, {'match': {'dtype': constraint}}]}}
            })
        if results['hits']['total'] > 0:
            return results['hits']['hits']
        return None


if __name__ == '__main__':
    e = Elastic(config['elastic']['server'],
                config['elastic']['entity_index_config'],
                config['dbpedia']['entities'],
                create_entity_index=False)
    print(e.search_ngram('bill finger', 'idx'))
