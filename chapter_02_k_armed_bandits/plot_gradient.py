import matplotlib.pyplot as plt
import pickle
import numpy as np

# Plot results
def plot(data:np.ndarray, 
        legends:list, 
        xlabel:str, 
        ylabel:str, 
        filename:str=None,
        fn=lambda:None,) -> None:

    fontdict={
        'fontsize': 12,
        'fontweight': 'bold',
    }

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(c='lightgray')
    plt.margins(0.02)

    # revers the loop for a better visualization
    colors = ['navy', 'lightblue', 'tomato', 'pink']
    for i in range(len(data)-1, -1, -1):
        # data[i] = uniform_filter(data[i])
        plt.plot(data[i], label=legends[i], linewidth=1.5, c=colors[i])
    
    # get rid of the top/right frame lines
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]: 
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)

    plt.tick_params(axis='both', labelsize=10)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel(ylabel, fontdict=fontdict)
    # plt.legend(loc=4, fontsize=13)
    fn()

    plt.text(500, 57, s="$\\alpha = 0.4$", c=colors[3], fontsize=14)
    plt.text(500, 28, s="$\\alpha = 0.4$", c=colors[1], fontsize=14)
    plt.text(900, 72, s="$\\alpha = 0.1$", c=colors[2], fontsize=14)
    plt.text(900, 52, s="$\\alpha = 0.1$", c=colors[0], fontsize=14)

    plt.text(770, 65, s="with baseline", c=colors[2], fontsize=12)
    plt.text(770, 42, s="without baseline", c=colors[0], fontsize=12)

    if not filename:
        plt.show()
    else:
        plt.savefig(f'./plots/{filename}')


def plot_result(optim_ratio:np.ndarray, 
                legends:list, 
                output_name:str=None):
    # Set tick labels
    fn = lambda: plt.yticks(np.arange(0, 100, 10), labels=[f'{val}%' for val in range(0, 100, 10)])
    plot(optim_ratio, 
            legends, 
            xlabel='Time step', 
            ylabel='% Optimal actions',
            filename=output_name,
            fn=fn)


if __name__ == "__main__":
    with open('./history/sga_record.pkl', 'rb') as f:
        history = pickle.load(f)
    
    optim_ratio = history['optim_acts_ratio'] * 100
    hyper_params = history['hyper_params']
    
    # plot_result(optim_ratio, hyper_params, output_name="example_2_5_sga.png")
    plot_result(optim_ratio, hyper_params, output_name=None)