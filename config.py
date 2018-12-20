import os

config = {
    'base_path': '/Users/hamid/workspace/DeepShallowParsingQA',
    'elastic': {
        'server': '127.0.0.1:9200',
        'entity_ngram_index_config': {
            'settings': {
                'max_ngram_diff': 10,
                'number_of_shards': 1,
                'number_of_replicas': 0,
                'analysis': {
                    'filter': {
                        'ngram_filter': {
                            'type': 'ngram',
                            'min_gram': 3,
                            'max_gram': 3,
                            "token_chars": ["letter", "digit"]
                        }
                    },
                    'analyzer': {
                        'ngram_analyzer': {
                            'type': 'custom',
                            'tokenizer': 'standard',
                            'filter': ['lowercase', 'ngram_filter']
                        }
                    }
                }
            },
            'mappings': {'resources': {
                'properties': {
                    'label': {
                        'type': 'text',
                        'analyzer': 'ngram_analyzer'
                    },
                    'key': {
                        'type': 'keyword'
                    },
                    'edge_count': {
                        'type': 'integer'
                    }
                }
            }}

        },
        'entity_whole_match_index_config': {
            'mappings': {'resources': {
                'properties': {
                    'label': {
                        'type': 'text',
                        "index_options": "docs",
                        "analyzer": "english"
                    },
                    'key': {
                        'type': 'keyword'
                    },
                    'edge_count': {
                        'type': 'integer'
                    }
                }
            }}
        },
        'relation_whole_match_index_config': {
            'mappings': {'resources': {
                'properties': {
                    'label': {
                        'type': 'text',
                        "index_options": "docs",
                        "analyzer": "english"
                    },
                    'key': {
                        'type': 'keyword'
                    },
                }
            }}
        }
    }
}
config['data_path'] = os.path.join(config['base_path'], 'data')
config['cache_path'] = os.path.join(config['data_path'], 'cache')
config['env_cache_path'] = os.path.join(config['cache_path'], 'env.cache')
config['glove_path'] = os.path.join(config['data_path'], 'glove/glove.840B.300d')
config['checkpoint_path'] = os.path.join(config['base_path'], 'checkpoint.chpt')

config['dbpedia'] = {'base_path': os.path.join(config['data_path'], 'dbpedia')}
config['dbpedia'] = {
    'entities': os.path.join(config['dbpedia']['base_path'], 'entities.json'),
    'relations': os.path.join(config['dbpedia']['base_path'], 'relations.json')
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
