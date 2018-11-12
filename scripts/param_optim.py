from config import config
from scripts.config_args import parse_args
from common.dataset.lc_quad import LC_QuAD
from common.model.runner import Runner

from sigopt import Connection


# Evaluate your model with the suggested parameter assignments
def evaluate_model(lc_quad, args, assignments):
    print(assignments)
    args.gamma = assignments['gamma']
    args.positive_reward = assignments['positive_reward']
    args.negetive_reward = assignments['negetive_reward']
    args.lr = assignments['lr']
    args.dropout = assignments['dropout']

    runner = Runner(lc_quad, args)
    runner.train(lc_quad, args)
    final_value = runner.test(lc_quad, args)

    return final_value


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

    args = parse_args()
    lc_quad = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      args.remove_entity, args.remove_stop_words)

    # Run the Optimization Loop until the Observation Budget is exhausted
    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()
        value = evaluate_model(lc_quad, args, suggestion.assignments)
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # Update the experiment object
        experiment = conn.experiments(experiment.id).fetch()
