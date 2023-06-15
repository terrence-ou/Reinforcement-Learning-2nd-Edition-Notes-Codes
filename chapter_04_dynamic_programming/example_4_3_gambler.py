import numpy as np
import matplotlib.pyplot as plt

# Get an array of options of the current state
def get_actions(stake:int) -> np.ndarray:
    max_stake = min(stake, 100 - stake)
    return np.arange(0, max_stake + 1)


# get values of the stats after the actions been taken
def get_next_values(V:np.ndarray, state:int, actions:np.ndarray) -> tuple:
    wins = V[state + actions]
    fails = V[state - actions]
    return (wins, fails)


# main value iteration algorithm
def value_iteration(prob_h:float=0.4,
                    theta:float=1e-5):
    
    global history
    global full_policy
    # Initialize the state-value function, with V(0) = 0 and V(100) = 1
    # such method allows for ignoring the reward function
    V = np.zeros(shape=(101,))
    V[-1] = 1

    policy = np.zeros(shape=(101,))

    # Value iteration
    while True:
        old_V = np.copy(V)
        for curr_state in range(1, 100):
            actions = get_actions(curr_state)[1:] # We ignore 0 action here
            wins, fails = get_next_values(V, curr_state, actions)
            values = prob_h * wins + (1 - prob_h) * fails # vectorization
            ops = np.where(values == values.max())[0]
            # update the value function
            V[curr_state] = values[ops[0]]
            # update policy
            policy[curr_state] = actions[ops[0]]
            full_policy[curr_state] = actions[ops]

        history.append(np.copy(V))

        # terminate if the maximum difference between two value functions 
        # is smaller than the threshold
        delta = np.abs(old_V - V).max()
        if delta < theta: break
    
    print('Total number of sweeps: ', len(history))
    return V, policy


# the plot visulization template
def visualization_setup(y_label: str):
    fontdict={
        'fontsize': 12,
        'fontweight': 'bold',
    }

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(c='lightgray', zorder=0)
    ticks = np.arange(0, 100, 10)[1:100]
    ticks = np.hstack([[1], ticks, [99]])
    plt.xticks(ticks)
    plt.margins(0.02)
    plt.xlabel('Capital', fontdict=fontdict)
    plt.ylabel(y_label, fontdict=fontdict)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]: 
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)


if __name__ == "__main__":

    history = [] # record history of state-value function
    full_policy = dict() # record full policy

    prob_head = 0.4 # For exercise 4.9, use the value 0.25 or 0.55
    theta = 1e-6

    V, pi = value_iteration(prob_h=prob_head, theta=theta)

    # Plot selected sweeps
    # revers the loop for a better visualization
    colors = ['cornflowerblue', 'tomato', 'lightseagreen', 'indigo']
    sweeps = [0, 1, 2, len(history)-1]

    visualization_setup(y_label='Value estimates')
    for i, sweep in enumerate(sweeps):
        plt.plot(np.arange(1, 100),
                history[sweep][1:100], 
                 label=f'sweep {sweep + 1}',
                 c=colors[i])
    plt.legend(loc=4)
    plt.show()
    # plt.savefig('./plots/example_4_3/values.png')


    visualization_setup(y_label="Final policy")
    # plot selected policy
    plt.step(np.arange(1, 100), pi[1:100], where='mid')
    plt.show()
    # plt.savefig('./plots/example_4_3/policy.png')

    # plot full policy
    visualization_setup(y_label="Full policy")
    for key, values in full_policy.items():
        keys = [key] * len(values)
        plt.scatter(keys, values, s=4.0, c='royalblue', zorder=2)
    plt.show()
    # plt.savefig('./plots/example_4_3/full_policy.png')