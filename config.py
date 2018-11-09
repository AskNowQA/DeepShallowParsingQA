import os

config = {
    'base_path': '/Users/hamid/workspace/DeepShallowParsingQA',
}
config['data_path'] = os.path.join(config['base_path'], 'data')
config['glove_path'] = os.path.join(config['data_path'], 'glove/glove.840B.300d')
config['checkpoint_path'] = os.path.join(config['base_path'], 'checkpoint.chpt')

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
