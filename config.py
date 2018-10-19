import os

config = {
    'glove_path': '/Users/hamid/workspace/SQG/learning/treelstm/data/glove/glove.840B.300d',
    'data_path': '/Users/hamid/workspace/DeepShallowParsingQA/data'
}

config['lc_quad'] = {
    'tiny': os.path.join(config['data_path'], 'lcquad', 'tiny.json'),
    'train': os.path.join(config['data_path'], 'lcquad', 'train-data.json')
}
