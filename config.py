import os

config = {
    'base_path': '/data/workspace/DeepShallowParsingQA',
    'http': {
        'timeout': 120
    },
    'dbpedia_kb': {
        # 'endpoint': 'http://dbpedia.org/sparql'
        'endpoint': 'http://sda01dbpedia:softrock@131.220.9.219/sparql'
    },
    'EARL': {
        'endpoint': 'http://sda.tech/earl/api/processQuery'
    },
    'elastic': {
        'server': 'iqa-elastic:9200',
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
            'mappings': {
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
            }

        },
        'entity_whole_match_index_config': {
            'mappings': {
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
            }
        },
        'relation_whole_match_index_config': {
            'mappings': {
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
            }
        }
    }
}
config['data_path'] = os.path.join(config['base_path'], 'data')
config['chk_path'] = os.path.join(config['data_path'], 'checkpoints')
config['cache_path'] = os.path.join(config['data_path'], 'cache')
config['env_cache_path'] = os.path.join(config['cache_path'], 'env.cache')
config['glove_path'] = os.path.join(config['data_path'], 'glove/glove.840B.300d')
config['EARL']['cache_path'] = os.path.join(config['cache_path'], 'earl.cache')

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
    'emb': os.path.join(config['lc_quad']['base_path'], 'dataset.emb'),
    'entity_one_hop': os.path.join(config['lc_quad']['base_path'], 'entity_one_hop.pk'),
}

config['qald_7_ml'] = {'base_path': os.path.join(config['data_path'], 'QALD')}
config['qald_7_ml'] = {
    'train': os.path.join(config['qald_7_ml']['base_path'], 'qald-7-train-multilingual.json'),
    'test': os.path.join(config['qald_7_ml']['base_path'], 'qald-7-test-multilingual.json'),
    'vocab': os.path.join(config['qald_7_ml']['base_path'], 'dataset.vocab'),
    'emb': os.path.join(config['qald_7_ml']['base_path'], 'dataset.emb'),
    'entity_one_hop': os.path.join(config['qald_7_ml']['base_path'], 'entity_one_hop.pk'),
}
