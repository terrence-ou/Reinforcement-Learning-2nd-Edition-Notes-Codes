import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Randomly sample mean reward for each action
    means = np.random.normal(size=(10, ))

    # Generate sample data based on normal distribution
    data = [np.random.normal(mean, 1.0, 2000) for mean in means]

    # Create violin plot
    plt.figure(figsize=(8, 6), dpi=150)
    plt.violinplot(dataset=data,
                   showextrema=False,
                   showmeans=False,
                   points=2000)

    # Draw mean marks
    for i, mean in enumerate(means):
        idx = i + 1
        plt.plot([idx - 0.3, idx + 0.3], [mean, mean],
                 c='black',
                 linewidth=1)
        plt.text(idx + 0.2, mean - 0.2, 
                 s=f"$q_*({idx})$",
                 fontsize=8)

    # Draw 0 dashed line
    plt.plot(np.arange(0, 12), np.zeros(12), 
                c='gray', 
                linewidth=0.5,
                linestyle=(5, (20, 10)))

    plt.tick_params(axis='both', labelsize=10)
    plt.xticks(np.arange(1, 11))

    # get rid of the frame
    for i, spine in enumerate(plt.gca().spines.values()):
        if i == 2: continue
        spine.set_visible(False)
        

    # Draw labels
    label_font = {
        'fontsize': 12,
        'fontweight': 'bold'
    }

    plt.xlabel('Action', fontdict=label_font)
    plt.ylabel('Reward distribution', fontdict=label_font)
    plt.margins(0)

    plt.tight_layout()
    # plt.show()
    plt.savefig('./plots/example_2_1.png')
