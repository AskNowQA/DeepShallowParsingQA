from tqdm import tqdm
import ujson as json
from config import config
from common.model.runner import Runner
from common.dataset.lc_quad import LC_QuAD
from common.dataset.qald_6_ml import Qald_6_ml
from common.dataset.qald_7_ml import Qald_7_ml
from scripts.config_args import parse_args

if __name__ == '__main__':

    args = parse_args()

    dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      False, args.remove_stop_words)
    # dataset = Qald_6_ml(config['qald_6_ml']['train'], config['qald_6_ml']['test'], config['qald_6_ml']['vocab'],
    #                     False, False)
    # dataset = Qald_7_ml(config['qald_7_ml']['train'], config['qald_7_ml']['test'], config['qald_7_ml']['vocab'],
    #                           False, False)

    runner = Runner(dataset, args)
    runner.load_checkpoint(checkpoint_filename = '/Users/hamid/workspace/DeepShallowParsingQA/data/checkpoints/lctmp')
    runner.environment.entity_linker = None
    runner.environment.relation_linker = None

    connecting_relations = False
    free_relation_match = False
    connecting_relation = False
    k = 10
    results = {}
    for idx, qarow in tqdm(enumerate(dataset.test_set)):
        result = runner.link(qarow.question, 0.1, k, connecting_relations, free_relation_match, connecting_relation, True)
        results[qarow.question] = result
    cache_path = './data/mdp+earl-lcquad.cache'
    with open(cache_path, 'w') as f:
        json.dump(results, f)
