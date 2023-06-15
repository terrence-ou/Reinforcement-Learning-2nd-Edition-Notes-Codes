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
    
    # Initialize the state-value function, with V(0) = 0 and V(100) = 1
    # such method allows for ignoring the reward function
    V = np.zeros(shape=(101,))
    V[-1] = 1

    policy = np.zeros(shape=(101,))
    history = None  # The history of value functions

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

        delta = np.abs(old_V - V).max()

        if type(history) != np.ndarray:
            history = V.reshape(1, -1)
        else:
            history = np.concatenate([history, V.reshape(1, -1)], axis=0)

        if delta < theta: break
    print('Total number of sweeps: ', history.shape[0])
    return V, policy


if __name__ == "__main__":
    V, pi = value_iteration(prob_h=0.4)
    plt.plot(V[1:100])
    plt.show()

    plt.step(np.arange(1, 100), pi[1:100])
    plt.show()

    history = np.zeros(shape=(100,))