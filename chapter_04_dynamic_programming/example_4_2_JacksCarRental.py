import gym_jcr
import gymnasium
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

MAX_CAPACITY = 20
TRANSFER_MAX = 5
GAMMA = 0.9


def get_transition_probs(P, s, policy):
    probs = P[:, s]
    return probs[range((MAX_CAPACITY + 1) ** 2), policy]

def get_rewards(R, s, policy):
    return R[s, policy[s] + TRANSFER_MAX]


if __name__ == "__main__":
    env = gymnasium.make('JacksCarRental-v0')
    R = env.reward
    P = env.transition
    print('Reward shape:', R.shape)
    print('Transition prob. shape:', P.shape)

    # 1. Initialization
    policy = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.int32)
    values = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.float64)
    converged = False


    while not converged:
        # 2. Policy evaluation
        while True:
            delta = 0
            old_values = np.copy(values)
            for s in range(env.nS):
                probs = get_transition_probs(P, s, policy) # shape: (441, )
                reward = get_rewards(R, s, policy)
                # new_val = rewards + GAMMA * probs * values
                new_val = probs * (reward + GAMMA * values)
                values[s] = new_val.sum()
            diff = abs(old_values - values).max()
            # print(diff)
            delta = max(delta, diff)
            if delta < 1e-6: 
                break
        print(values.mean())

        # 3. Policy improvement
        policy_stable = True
        for s in range(env.nS):
            break

        break