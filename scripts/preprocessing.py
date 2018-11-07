from common.vocab import Vocab
from config import config
import pickle as pk
import torch

file_path = '/Users/hamid/workspace/DeepShallowParsingQA/data/lcquad/relations.pickle'
with open(file_path, 'rb') as f_h:
    rel2id = pk.load(f_h, encoding='latin1')

vocab = Vocab(filename=config['glove_path'] + '.vocab')

## Need to fix cases where there are non-alphabet chars in the label
for item_id, item in rel2id.items():
    idxs = [vocab.getIndex(word) for word in item[2]]
    idxs = [id for id in idxs if id is not None]
    idxs = torch.LongTensor(idxs)
    if len(item) >= 6:
        item[5] = idxs
    else:
        item.append(idxs)

with open(file_path, 'wb') as f_h:
    pk.dump(rel2id, f_h)
