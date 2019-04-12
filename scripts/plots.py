import matplotlib
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'normal',
        # 'weight' : 'bold',
        'size': 16}


# matplotlib.rc('font', **font)


def MRR_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    entity_mrrs = [0.7158333333333332, 0.7227777777777777, 0.7261527777777779, 0.7279861111111111,
                   0.7286527777777777, 0.729938492063492, 0.7302509920634921, 0.7305287698412698, 0.73097876984127,
                   0.7312818001443002]
    earl_entity_mrrs = [0.5439499999999999, 0.5552833333333331, 0.559283333333333, 0.5611999999999999,
                        0.5623666666666666,
                        0.5632476190476189, 0.5633934523809521, 0.5635601190476187, 0.5640601190476188,
                        0.5641964826839824]
    relation_mrrs = [0.3900833333333334, 0.40444444444444444, 0.4124305555555555, 0.4171277777777778,
                     0.42078055555555555,
                     0.4257253968253968, 0.42737916666666664, 0.4283278439153439, 0.43058062169312167,
                     0.44391963684463687]
    earl_relation_mrrs = [0.37835833333333335, 0.3828805555555555, 0.3863388888888888, 0.38698888888888877,
                          0.38748055555555544, 0.38771865079365075, 0.3879415674603174, 0.3881082341269841,
                          0.3881332341269841, 0.3881938401875901]
    ax.plot(range(1, len(entity_mrrs) + 1), entity_mrrs, marker='x', color='green', label='MDP-Parser')
    ax.plot(range(1, len(earl_entity_mrrs) + 1), earl_entity_mrrs, marker='o', color='black', label='EARL')
    plt.xticks(range(1, len(entity_mrrs) + 1))
    plt.legend(loc='center right')
    plt.xlabel('k=1..10')
    plt.ylabel('Mean Reciprocal Rank (MRR)')
    fig.tight_layout()
    plt.savefig('mrr_entity_k.png')
    plt.show()
    plt.close()
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(range(1, len(relation_mrrs) + 1), relation_mrrs, marker='x', color='blue', label='MDP-Parser')
    ax.plot(range(1, len(earl_relation_mrrs) + 1), earl_relation_mrrs, marker='o', color='black', label='EARL')
    plt.xticks(range(1, len(entity_mrrs) + 1))
    plt.xlabel('k=1..10')
    plt.ylabel('Mean Reciprocal Rank (MRR)')
    plt.legend(loc='center right')
    fig.tight_layout()
    plt.savefig('mrr_relation_k.png')
    plt.show()


def error_analysis_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = 'Linker Failure', 'Missing/No Relation Span', 'Incomplete Entity Span', 'Others'
    ind = np.arange(len(labels))
    width = 0.3
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
        ax.text(v - 0.04, -0.05 + i - width, "{:.2f}".format(v), color='white', fontweight='bold')

    for i, v in enumerate(earl):
        ax.text(v - 0.04, -0.05 + i, "{:.2f}".format(v), color='white', fontweight='bold')

    for i, v in enumerate(falcon):
        ax.text(v - 0.04, -0.05 + i + width, "{:.2f}".format(v), color='white', fontweight='bold')

    plt.gca().set_yticks(ind)
    plt.gca().set_yticklabels(labels)
    ax.set_xlabel('Percentage')
    ax.legend()
    fig.tight_layout()
    plt.savefig('error_analysis.png')
    plt.show()


if __name__ == '__main__':
    # MRR_plot()
    error_analysis_plot()
