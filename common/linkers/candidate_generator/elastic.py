from elasticsearch import Elasticsearch
import ujson as json
from tqdm import tqdm
import re


class Elastic:
    def __init__(self, server):
        self.es = Elasticsearch(hosts=[server])

    def create_index(self, index_config, input_path, index_name='idx'):
        batch_size = 100000
        delete_index = True

        type_name = 'resources'
        bulk_data = []
        counter = 0
        uris = []
        with open(input_path, 'r', encoding='utf-8') as file_handler:
            for line in tqdm(file_handler):
                json_object = json.loads(line)['_source']
                uri = json_object['uri']
                if 'http://dbpedia.org/' in uri:
                    dtype = 'uri'
                else:
                    dtype = 'literal'
                label = ''
                if 'dbpediaLabel' in json_object:
                    label = json_object['dbpediaLabel']
                elif 'wikidataLabel' in json_object:
                    label = json_object['wikidataLabel']
                elif 'mergedLabel' in json_object:
                    label = json_object['mergedLabel']

                if len(label) <= 2 or len(label) > 70:
                    continue
                label = label.lower()
                data_dict = {'key': uri,
                             'dtype': dtype,
                             'label': label
                             }
                if 'edgecount' in json_object:
                    data_dict['edge_count'] = json_object['edgecount']
                op_dict = {"index": {"_index": index_name, "_type": type_name}}
                bulk_data.append(op_dict)
                bulk_data.append(data_dict)
                if uri not in uris:
                    label = uri[uri.rindex('/') + 1:]
                    label = re.sub(r"([A-Z])", r" \1", label).replace('_', ' ')
                    data_dict = {'key': uri,
                                 'dtype': dtype,
                                 'label': label
                                 }
                    if 'edgecount' in json_object:
                        data_dict['edge_count'] = json_object['edgecount']
                    op_dict = {"index": {"_index": index_name, "_type": type_name}}
                    bulk_data.append(op_dict)
                    bulk_data.append(data_dict)
                    uris.append(uri)

                if counter > 0 and counter % batch_size == 0:
                    self.bulk_indexing(index_name=index_name,
                                       delete_index=delete_index,
                                       index_config=index_config,
                                       bulk_data=bulk_data)
                    bulk_data = []
                    delete_index = False
                counter += 1
            if len(bulk_data) > 0:
                self.bulk_indexing(index_name=index_name,
                                   delete_index=delete_index,
                                   index_config=index_config,
                                   bulk_data=bulk_data)

    def bulk_indexing(self, index_name, delete_index, index_config, bulk_data):
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
        res = self.es.bulk(index=index_name, body=bulk_data, refresh=True, request_timeout=60)
        print("bulk indexing done")
        return res

    def search_index(self, text, index, constraint=None, size=10):
        output = []
        if constraint is None:
            results = self.es.search(index=index, doc_type='resources', size=size, body={
                'query': {'match': {'label': text, }}
            })
        else:
            results = self.es.search(index=index, doc_type='resources', size=size, body={
                'query': {'bool': {'must': [{'match': {'label': text}}, {'match': {'dtype': constraint}}]}}
            })
        if results['hits']['total'] > 0:
            if 'relation' in index:
                output = [[item['_source']['key'], item['_source']['key'][item['_source']['key'].rindex('/') + 1:]] for
                          item in results['hits']['hits']]
            else:
                output = [[item['_source']['key'], item['_source']['label']] for item in results['hits']['hits']]
        else:
            print(results)
        return output

    def search_term(self, text, index, size=10):
        output = []
        results = self.es.search(index=index, doc_type='resources', size=size, body={
            'query': {'term': {'key': text, }}
        })

        if results['hits']['total'] > 0:
            output = [[item['_source']['key'], item['_source']['label']] for item in results['hits']['hits']]
        else:
            print(results)
        return output
