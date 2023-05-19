import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple


history = namedtuple('history', ['bounds', 'data'])
algos = ['e_greedy', 'gradient', 'ucb', 'oiv']

if __name__ == '__main__':

    with open('./history/summary.pkl', 'rb') as f:
        histories = pickle.load(f)
        coords = [[0.95, 1.55], [6.5, 1.45], [3, 1.82], [8.5, 1.82]]
        legend_loc = 2
        filename='./plots/example_2_6_summary.png'


    # with open('./history/exercise_2_6.pkl', 'rb') as f:
    #     histories = pickle.load(f)
    #     coords = [[2.5, 6.0], [7.0, 3.5], [7.5, 5.0], [6.5, 5.7]]
    #     legend_loc = 0
    #     filename = './plots/exercise_2_6.png'

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
    plt.legend(loc=legend_loc, fontsize=12, title='Hyper Param.')
    plt.xlabel('Hyper parameter value', fontdict=fontdict)
    plt.ylabel('Average reward over first 1000 steps', 
                fontdict=fontdict)
    
    plt.text(*coords[0], '$\epsilon$-greedy', c=colors[0], fontsize=12)
    plt.text(*coords[1], 'gradient\nbandit', c=colors[1], fontsize=12, horizontalalignment='center')
    plt.text(*coords[2], 'UCB', c=colors[2], fontsize=12)
    plt.text(*coords[3], 'greedy with\noptimistic\ninitializatio\n$\\alpha=0.1$', c=colors[3], fontsize=12, horizontalalignment='center')

    # plt.show()
    plt.savefig(filename)