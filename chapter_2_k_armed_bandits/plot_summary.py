import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple


history = namedtuple('history', ['bounds', 'data'])
algos = ['e_greedy', 'gradient', 'ucb', 'oiv']

if __name__ == '__main__':

    with open('./history/summary.pkl', 'rb') as f:
        histories = pickle.load(f)

    x_ticks = histories['x_labels']    

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(c='lightgray')
    plt.margins(0.02)

    fontdict={
        'fontsize': 12,
        'fontweight': 'bold',
    }

    legends = ['$\epsilon$', '$\\alpha$', '$c$', '$Q_0$']
    colors = ['tomato', 'mediumseagreen', 'steelblue', 'orchid']

    for i, key in enumerate(algos):
        record = histories[key]
        bounds = record.bounds
        data = record.data

        plt.plot(np.arange(bounds[0], bounds[1]), data, label=legends[i], c=colors[i])
    

    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]: 
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)

    plt.tick_params(axis='both', labelsize=10)
    plt.xticks(np.arange(10), x_ticks)
    
    # x labels
    plt.legend(loc=2, fontsize=12, title='Hyper Param.')
    plt.xlabel('Hyper parameter value', fontdict=fontdict)
    plt.ylabel('Average reward over first 1000 steps', 
                fontdict=fontdict)
    
    plt.text(0.95, 1.55, '$\epsilon$-greedy', c=colors[0], fontsize=12)
    plt.text(6.5, 1.45, 'gradient\nbandit', c=colors[1], fontsize=12, horizontalalignment='center')
    plt.text(3, 1.82, 'UCB', c=colors[2], fontsize=12)
    plt.text(7.5, 1.58, 'greedy with\noptimistic\ninitializatio\n$\epsilon=0.1$', c=colors[3], fontsize=12, horizontalalignment='center')

    # plt.show()
    plt.savefig('./plots/example_2_6_summary.png')