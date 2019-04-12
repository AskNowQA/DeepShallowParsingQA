from common.linkers.candidate_generator.elastic import Elastic
from common.dataset.lc_quad import LC_QuAD
from common.vocab import Vocab
from common.dataset.container.uri import URI
from config import config
import pickle as pk
import ujson as json
from tqdm import tqdm
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create ElasticSearch Index')
    parser.add_argument('--create_index', dest='create_index', action='store_true')
    parser.add_argument('--index_name', default='idx')
    parser.add_argument('--search', default='search')
    parser.add_argument('--size', default=10)
    args = parser.parse_args()

    e = Elastic(config['elastic']['server'])
    if args.create_index:
        if 'entit' in args.index_name:
            # index_config =config['elastic']['entity_ngram_index_config']
            index_config = config['elastic']['entity_whole_match_index_config']
            e.create_index(index_config,
                           config['dbpedia']['entities'],
                           index_name=args.index_name)
            dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                              False, False)
            bulk_data = []
            for qa_row in dataset.train_set + dataset.test_set:
                for entity in qa_row.sparql.entities:
                    result = e.search_term(entity.raw_uri, args.index_name)
                    result = [item for item in result if item[0] == entity.raw_uri]
                    if len(result) == 0:
                        print(entity.raw_uri)
                        data_dict = {'key': entity.raw_uri,
                                     'dtype': 'uri',
                                     'label': entity.label.replace('_', ' ')
                                     }
                        op_dict = {"index": {"_index": args.index_name, "_type": 'resources'}}
                        bulk_data.append(op_dict)
                        bulk_data.append(data_dict)
            e.bulk_indexing(args.index_name, delete_index=False, index_config=index_config, bulk_data=bulk_data)

        elif 'relation' in args.index_name:
            e.create_index(config['elastic']['relation_whole_match_index_config'],
                                    config['dbpedia']['relations'],
                                    index_name=args.index_name)
            vocab = Vocab(filename=config['lc_quad']['vocab'], data=['<ent>', '<num>'])

            coded_labels = {}
            max_length = 3
            with open(config['dbpedia']['relations'], 'r', encoding='utf-8') as file_handler:
                for line in tqdm(file_handler):
                    json_object = json.loads(line)['_source']
                    uri = json_object['uri']
                    if 'http://dbpedia.org/' in uri:
                        uri = URI(uri)
                        if uri.raw_uri not in coded_labels:
                            idxs = vocab.convertToIdx(uri.tokens, '')[:max_length]
                            length = len(idxs)
                            if len(idxs) < max_length:
                                idxs = idxs + [0] * (max_length - len(idxs))
                            coded_labels[uri.raw_uri] = [torch.LongTensor(idxs), length]

            with open(config['dbpedia']['relations'] + '.coded', 'wb') as file_handler:
                pk.dump(coded_labels, file_handler)

    print(e.search_index(args.search, args.index_name, size=args.size))
