from common.linkers.candidate_generator.elastic import Elastic
from config import config
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create ElasticSearch Index')
    parser.add_argument('--create_index', dest='create_index', action='store_true')
    parser.add_argument('--search', default='search')
    args = parser.parse_args()

    e = Elastic(config['elastic']['server'],
                config['elastic']['entity_index_config'],
                config['dbpedia']['entities'],
                create_entity_index=args.create_index)
    print(e.search_ngram(args.search, 'idx'))
