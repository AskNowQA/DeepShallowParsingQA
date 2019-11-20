import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import ujson as json

font = {'family': 'normal',
        # 'weight' : 'bold',
        'size': 12}

matplotlib.rc('font', **font)

labels = {'lstm': 'LSTM', 'bilstm': 'Bi-LSTM', 'nn': 'Fully-Connected'}
plot_colors = {'lstm': 'red', 'bilstm': 'blue', 'nn': 'green'}


def policy_plot(avg=True, dataset='lcquad'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    with open('./data/eval-{}.json'.format(dataset), 'rt') as json_file:
        eval_results = json.load(json_file)

    policies = set([name[len(dataset) + 1:-6] for name in eval_results.keys() if '_' not in name])  #
    bs = set([name[-4:-3] for name in eval_results.keys() if '+' not in name])  #
    start = 1

    mrrs = {
        policy: {str(b): np.array([eval_results['{}-{}-b{}-i{}'.format(dataset, policy, b, i)] for i in range(1, 6)])
                 for b in bs} for policy in policies}

    n_squared = np.sqrt(len(bs))
    for policy in policies:
        print(policy)
        x = list(range(start, len(bs)))
        y = np.array([np.mean(mrrs[policy][str(b)]) for b in range(0, len(bs))])[start:]
        error = np.array([np.mean(np.std(mrrs[policy][str(b)], axis=0)) for b in range(0, len(bs))])[start:]
        error = error / n_squared
        ax.plot(x, y, label=labels[policy], color=plot_colors[policy])
        ax.fill_between(x, y - error, y + error, alpha=0.5, color=plot_colors[policy])  # , label=policy)
        print(error)
        print(y)

    plt.xticks(range(start, len(bs)), range(start, len(bs)))
    # plt.yticks(np.arange(0.3, 0.6, step=0.1))
    ax.legend(loc='lower right', ncol=3, bbox_to_anchor=(1, 1.02), borderaxespad=0.)
    plt.xlabel('Number of prev. and next words (h)')
    plt.ylabel('Mean Reciprocal Rank (MRR), K=1')
    fig.tight_layout()
    plt.savefig('./figs/policy-{0}.png'.format(dataset))
    plt.close()


def MRR_plot(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    file_name = './data/mrr-mdp-{}.json'.format(dataset)
    with open(file_name, 'rt') as json_file:
        mdp_results = json.load(json_file)

    file_name = './data/mrr-mdp+earl-{}.json'.format(dataset)
    with open(file_name, 'rt') as json_file:
        mdp_earl_results = json.load(json_file)

    file_name = './data/mrr-earl-{}.json'.format(dataset)
    with open(file_name, 'rt') as json_file:
        earl_results = json.load(json_file)

    mdp_entity_mrrs = [mdp_results[item][0] for item in mdp_results][:10]
    mdp_earl_entity_mrrs = [mdp_earl_results[item][0] for item in mdp_earl_results][:10]
    earl_entity_mrrs = [earl_results[item][0] for item in earl_results][:10]
    mdp_relation_mrrs = [mdp_results[item][1] for item in mdp_results][:10]
    mdp_earl_relation_mrrs = [mdp_earl_results[item][1] for item in mdp_earl_results][:10]
    earl_relation_mrrs = [earl_results[item][1] for item in earl_results][:10]
    ax.plot(range(1, len(mdp_entity_mrrs) + 1), mdp_entity_mrrs, marker='x', color='green', label='MDP-Parser')
    ax.plot(range(1, len(mdp_earl_entity_mrrs) + 1), mdp_earl_entity_mrrs, marker='^', color='blue',
            label='EARL+MDP-Parser')
    ax.plot(range(1, len(earl_entity_mrrs) + 1), earl_entity_mrrs, marker='o', color='black', label='EARL')
    plt.xticks(range(1, len(mdp_entity_mrrs) + 1))
    plt.legend(loc='center right')
    plt.xlabel('k')
    plt.ylabel('Mean Reciprocal Rank (MRR)')
    fig.tight_layout()
    plt.savefig('./figs/mrr_entity_k.png')
    # plt.show()
    plt.close()
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(range(1, len(mdp_relation_mrrs) + 1), mdp_relation_mrrs, marker='x', color='green', label='MDP-Parser')
    ax.plot(range(1, len(mdp_earl_relation_mrrs) + 1), mdp_earl_relation_mrrs, marker='^', color='blue',
            label='EARL+MDP-Parser')
    ax.plot(range(1, len(earl_relation_mrrs) + 1), earl_relation_mrrs, marker='o', color='black', label='EARL')
    plt.xticks(range(1, len(mdp_entity_mrrs) + 1))
    plt.xlabel('k')
    plt.ylabel('Mean Reciprocal Rank (MRR)')
    plt.legend(loc='center right')
    fig.tight_layout()
    plt.savefig('./figs/mrr_relation_k.png')
    # plt.show()


def error_analysis_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = 'Linker Failure', 'Missing/No Relation Span', 'Incomplete Entity Span', 'Others'
    ind = np.arange(len(labels))
    width = 0.2
    us = [64, 35, 19, 12]
    sum_ = sum(us)
    us = [item / sum_ for item in us]
    earl = [37, 11, 33, 10]
    sum_ = sum(earl)
    earl = [item / sum_ for item in earl]
    falcon = [24, 6, 17, 11]
    sum_ = sum(falcon)
    falcon = [item / sum_ for item in falcon]
    ax.barh(ind - width, us, width, label='MDP-Parser')
    ax.barh(ind, earl, width, color='gray', label='EARL')
    ax.barh(ind + width, falcon, width, color='brown', label='Falcon')

    for i, v in enumerate(us):
        ax.text(v - 0.06, -0.05 + i - width, "{:.2f}".format(v), color='white', fontweight='bold')

    for i, v in enumerate(earl):
        ax.text(v - 0.06, -0.05 + i, "{:.2f}".format(v), color='white', fontweight='bold')

    for i, v in enumerate(falcon):
        ax.text(v - 0.06, -0.05 + i + width, "{:.2f}".format(v), color='white', fontweight='bold')

    plt.gca().set_yticks(ind)
    plt.gca().set_yticklabels(labels)
    ax.set_xlabel('Percentage')
    ax.legend()
    fig.tight_layout()
    plt.savefig('error_analysis.png')
    plt.show()


if __name__ == '__main__':
    MRR_plot('lcquad')
    # b_plot()
    # policy_plot(dataset='lcquad')
    # error_analysis_plot()
