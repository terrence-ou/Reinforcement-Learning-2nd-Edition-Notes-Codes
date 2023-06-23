from matplotlib import pyplot as plt
import numpy as np

# Plot state-value function
def plot_surface(V, title=None, savefig=False, file_path_name=None):

    fontdict = {'fontsize': 11, 
                # 'fontweight': 'bold'
                }
    
    title_fontdict = {'fontsize': 12, 
                      'fontweight': 'bold'}

    plt.rcParams['grid.color'] = 'lightgray'

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                        figsize=(8, 6), dpi=150 if not savefig else 300)
    ax.set_box_aspect((1, 1, 0.3)) # set 3d plot size
    
    # Organizing dataset
    V = V[12:22, 1:]
    Y = np.arange(V.shape[0])
    X = np.arange(V.shape[1])
    X, Y = np.meshgrid(X, Y)

    # Plot value surface
    ax.plot_surface(X, Y, V, 
                    cmap='viridis',
                    rstride=1,
                    cstride=1,
                    linewidth=0.5,
                    edgecolor='white',
                    alpha=0.9
                    )

    if title:
        ax.set_title(title, fontdict=title_fontdict)

    ax.set_xticks(range(0, 10), ['A'] + [f'{i}' for i in range(2, 11)])
    ax.set_yticks(range(0, 10), range(12, 22))
    ax.set_zticks([-1.0, 0, 1.0])

    ax.set_xlabel('Dealer Showing', fontdict=fontdict)
    ax.set_ylabel('Player Sum', fontdict=fontdict)
    ax.view_init(elev=33, azim=-50)
    
    # Remove axis background colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if not savefig:
        plt.show()
    else:
        plt.savefig(file_path_name, bbox_inches='tight')



def plot_policy(policy):
    pass