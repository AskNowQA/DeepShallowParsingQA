from config import config
from common.word_vectorizer.glove import Glove
from common.dataset.lc_quad import LC_QuAD
from common.model.agent import Agent
from common.model.policy import Policy
from common.model.environment import Environment
from common.model.runner import Runner
from common.linkers.orderedLinker import OrderedLinker
from common.linkers.stringSimilaritySorter import StringSimilaritySorter

import numpy as np
import torch
from tqdm import tqdm

from sigopt import Connection
from sigopt.examples import franke_function


# Evaluate your model with the suggested parameter assignments
def evaluate_model(dataset, word_vectorizer, linker, assignments):
    print(assignments)
    policy_network = Policy(vocab_size=dataset.vocab.size(),
                            emb_size=word_vectorizer.word_size,
                            input_size=word_vectorizer.word_size * 2 + 1,
                            hidden_size=int(word_vectorizer.word_size * 4),
                            output_size=2,
                            dropout_ratio=0.5)
    agent = Agent(number_of_relations=2,
                  gamma=assignments['gamma'],
                  policy_network=policy_network,
                  policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=assignments['lr']))
    env = Environment(linker=linker,
                      positive_reward=assignments['positive_reward'],
                      negative_reward=assignments['negetive_reward'])
    runner = Runner(environment=env, agent=agent)
    total_reward = []
    last_idx = 0
    e = 0.001
    final_value = 0
    for i in tqdm(range(2000)):
        for idx, doc in enumerate(dataset.dataset):
            env.set_target(doc[2])
            total_reward.append(runner.step(dataset.coded_corpus[idx], e))

        if i % 100 == 0:
            final_value = np.sum(np.array(total_reward[last_idx:]) > 0) / len(total_reward[last_idx:])
            print(np.mean(total_reward[last_idx:]), final_value, e)
            last_idx = len(total_reward)

        # e = 1 / (i / 100 + 1)
        # if e < 0.1:
        #     e = 0.1

    return final_value
    # return franke_function(assignments['x'], assignments['y'])


if __name__ == '__main__':
    conn = Connection(client_token="LVUPKMBSCXYRGLMEQGEZHUIURKDSRJZHWRRWCYRAFLPQJJFN")

    experiments = conn.experiments().fetch()
    if len(experiments.data) > 0:
        experiment = experiments.data[0]
    # experiment = conn.experiments().create(
    #     name='RL Optimization (Python)',
    #     # Define which parameters you would like to tune
    #     parameters=[
    #         dict(name='dropout', type='double', bounds=dict(min=0.0, max=1.0)),
    #         dict(name='gamma', type='double', bounds=dict(min=0.0, max=1.0)),
    #         dict(name='lr', type='double', bounds=dict(min=0.0000001, max=0.1)),
    #         dict(name='positive_reward', type='int', bounds=dict(min=1, max=10)),
    #         dict(name='negetive_reward', type='int', bounds=dict(min=-10, max=0)),
    #     ],
    #     metrics=[dict(name='function_value')],
    #     parallel_bandwidth=1,
    #     # Define an Observation Budget for your experiment
    #     observation_budget=5,
    # )
    # print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

    lc_quad = LC_QuAD(config['lc_quad']['tiny'], config['lc_quad']['vocab'])
    word_vectorizer = Glove(lc_quad, config['glove_path'], config['lc_quad']['emb'])
    linker = OrderedLinker(sorter=StringSimilaritySorter(),
                           rel2id_path=config['lc_quad']['rel2id'],
                           core_chains_path=config['lc_quad']['core_chains'],
                           dataset=lc_quad)

    # Run the Optimization Loop until the Observation Budget is exhausted
    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()
        value = evaluate_model(lc_quad, word_vectorizer, linker, suggestion.assignments)
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # Update the experiment object
        experiment = conn.experiments(experiment.id).fetch()
