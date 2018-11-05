import os

config = {
    'glove_path': '/Users/hamid/workspace/SQG/learning/treelstm/data/glove/glove.840B.300d',
    'data_path': '/Users/hamid/workspace/DeepShallowParsingQA/data'
}

config['lc_quad'] = {
    'base_path': os.path.join(config['data_path'], 'lcquad')
}
config['lc_quad'] = {
    'tiny': os.path.join(config['lc_quad']['base_path'], 'tiny.json'),
    'train': os.path.join(config['lc_quad']['base_path'], 'train-data.json'),
    'test': os.path.join(config['lc_quad']['base_path'], 'test-data.json'),
    'rel2id': os.path.join(config['lc_quad']['base_path'], 'relations.pickle'),
    'core_chains': os.path.join(config['lc_quad']['base_path'], 'id_big_data.json'),
    'vocab': os.path.join(config['lc_quad']['base_path'], 'dataset.vocab'),
    'emb': os.path.join(config['lc_quad']['base_path'], 'dataset.emb')
}
