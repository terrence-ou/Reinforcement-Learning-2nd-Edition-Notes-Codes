import numpy as np
import matplotlib.pyplot as plt

def get_actions(stake:int) -> np.array:
    max_stake = min(stake, 100 - stake)
    return np.arange(0, max_stake + 1)

def get_next_states(V:np.array, state:int, actions:np.array) -> tuple:
    wins = V[(state + actions)]
    fails = V[(state - actions)]
    return (wins, fails)
    

if __name__ == "__main__":
    # Initialize the state-value function, with V(0) = 0 and V(100) = 1
    # such method allows for ignoring the reward function
    V = np.zeros(shape=(101,))
    V[-1] = 1
    # Probability of head
    prob_h = 0.4
    # threshold of convergence
    theta = 1e-3
    history = None

    # Value iteration
    while True:
        old_V = np.copy(V)
        for curr_state in range(1, 100):
            actions = get_actions(curr_state)
            wins, fails = get_next_states(V, curr_state, actions)
            V[curr_state] = np.max(prob_h * wins + (1 - prob_h) * fails)
        
        delta = np.abs(old_V - V).max()

        if type(history) != np.array:
            history = V
        else:
            history = np.concatenate(history, V)

        if delta < theta: break

    print('Total number of sweeps: ', history.shape[0])