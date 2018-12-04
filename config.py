import os

config = {
    'base_path': '/Users/hamid/workspace/DeepShallowParsingQA',
    'elastic': {
        'server': '127.0.0.1:9200',
        'entity_index_config': {
            'settings': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
                'analysis': {
                    'filter': {
                        'trigrams_filter': {
                            'type': 'ngram',
                            'min_gram': 3,
                            'max_gram': 3
                        }
                    },
                    'analyzer': {
                        'trigrams': {
                            'type': 'custom',
                            'tokenizer': 'standard',
                            'filter': ['lowercase', 'trigrams_filter']
                        }
                    }
                }
            },
            'mappings': {'resources': {
                'properties': {
                    'label': {
                        'type': 'text',
                        'analyzer': 'trigrams'
                    },
                    'wikidata_label': {
                        'type': 'text',
                        'analyzer': 'trigrams'
                    },
                    'edge_count': {
                        'type': 'integer'
                    }
                }
            }}

        }
    }
}
config['data_path'] = os.path.join(config['base_path'], 'data')
config['glove_path'] = os.path.join(config['data_path'], 'glove/glove.840B.300d')
config['checkpoint_path'] = os.path.join(config['base_path'], 'checkpoint.chpt')

config['dbpedia'] = {'base_path': os.path.join(config['data_path'], 'dbpedia')}
config['dbpedia'] = {
    'entities': os.path.join(config['dbpedia']['base_path'], 'entities.json')
}

config['lc_quad'] = {'base_path': os.path.join(config['data_path'], 'lcquad')}
config['lc_quad'] = {
    'tiny': os.path.join(config['lc_quad']['base_path'], 'tiny.json'),
    'train': os.path.join(config['lc_quad']['base_path'], 'train-data.json'),
    'test': os.path.join(config['lc_quad']['base_path'], 'test-data.json'),
    'rel2id': os.path.join(config['lc_quad']['base_path'], 'relations.pickle'),
    'core_chains': os.path.join(config['lc_quad']['base_path'], 'id_big_data.json'),
    'vocab': os.path.join(config['lc_quad']['base_path'], 'dataset.vocab'),
    'emb': os.path.join(config['lc_quad']['base_path'], 'dataset.emb')
}
