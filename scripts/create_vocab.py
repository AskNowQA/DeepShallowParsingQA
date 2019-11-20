import re
import os
from tqdm import tqdm
import ujson as json
import pickle as pk
import torch

from common.dataset.container.uri import URI
from config import config
from common.vocab import Vocab
from common.word_vectorizer.glove import Glove
from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_7_ml import Qald_7_ml
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.simple_dbpedia_qa import SimpleDBpediaQA

if __name__ == '__main__':
    print('Create Vocab')
    datasets = [Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
                          False, False),
                Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
                          False, False),
                LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                        False, False)]
    if not os.path.exists(config['vocab']):
        #         SimpleDBpediaQA(config['SimpleDBpediaQA']['train'], config['SimpleDBpediaQA']['test'],
        #                         config['SimpleDBpediaQA']['vocab'],
        #                         False, False)]
        vocab = set()
        for dataset in tqdm(datasets):
            lines = dataset.corpus
            for tokens in lines:
                vocab |= set(tokens)
            if dataset.one_hop is not None:
                for entity, uris in dataset.one_hop.items():
                    for idx in range(len(uris)):
                        uri = URI(uris[idx][0])
                        vocab |= set(uri.tokens)
            print(len(vocab))

        with open(config['dbpedia']['relations'], 'r', encoding='utf-8') as file_handler:
            for line in tqdm(file_handler):
                json_object = json.loads(line)['_source']
                uri = json_object['uri']
                if 'http://dbpedia.org/' in uri:
                    uri = URI(uri)
                    vocab |= set(uri.tokens)
        print(len(vocab))
        vocab_list = [URI.normalize(word) for word in vocab]
        vocab = set([word for words in vocab_list for word in words])
        with open(config['vocab'], 'w', encoding='utf-8') as f:
            for token in sorted(vocab):
                f.write(token + '\n')

    vocab = Vocab(config['vocab'], data=['<ukn>', '<ent>', '<num>'])
    word_vectorizer = Glove(vocab, config['glove_path'], config['emb'])

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

    for dataset in tqdm(datasets):
        if dataset.one_hop is not None:
            for entity, uris in dataset.one_hop.items():
                for idx in range(len(uris)):
                    uri = URI(uris[idx][0])
                    idxs = vocab.convertToIdx(uri.tokens, '')[:max_length]
                    length = len(idxs)
                    if len(idxs) < max_length:
                        idxs = idxs + [0] * (max_length - len(idxs))
                    coded = torch.LongTensor(idxs)
                    if len(uris[idx]) == 2:
                        uris[idx].extend([coded, length])
                    else:
                        uris[idx][2] = coded
                        uris[idx][3] = length
            with open(dataset.config['entity_one_hop'], 'wb') as f:
                pk.dump(dataset.one_hop, f)
