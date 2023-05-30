import gym_jcr
import gymnasium
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

MAX_CAPACITY = 20
TRANSFER_MAX = 5
GAMMA = 0.9

env = gymnasium.make('JacksCarRental-v0')
Reward = env.reward
P = env.transition

def get_transition_kernel_for_policy(policy):
    return P[:, range((MAX_CAPACITY + 1) ** 2), policy + TRANSFER_MAX]


def get_P_reward_for_policy(policy):
    P_pi = get_transition_kernel_for_policy(policy)
    return P_pi, R[range((MAX_CAPACITY + 1) ** 2), policy + TRANSFER_MAX]

# Policy evaluation
def evaluate_policy_by_iteration(policy, values = np.zeros((MAX_CAPACITY+1)**2)):
    P_pi, reward = get_P_reward_for_policy(policy)
    converged = False
    while not converged:
        new_values = reward + GAMMA * np.dot (P_pi.T, values)
        if np.allclose(new_values, values, rtol=1e-07, atol=1e-10):
            converged = True
        values = new_values
    return values

# Greedy policy
def greedy_improve(values, P=P):
    P_ = P.transpose(1, 0, 2) # we used the model for improvement
    all_pure_states = np.eye((MAX_CAPACITY+1)**2)
    new_states = np.dot(all_pure_states, P_) 
    q = np.dot(new_states.transpose(2,1,0), values) 
    q = q.T + Reward
    policy_indices = np.argmax(q, axis=1)
    policy = policy_indices - TRANSFER_MAX
    return policy

# Policy improvement
def improve_policy(policy):
    values = evaluate_policy_by_iteration(policy)
    not_converged = True
    while not_converged:
        print (values.mean())
        new_policy = greedy_improve(values)
        #new_values = evaluate_policy(new_policy)
        new_values = evaluate_policy_by_iteration(new_policy, values)
        if np.allclose(new_values, values, rtol=1e-02):
            not_converged = False
        values = new_values 
    return new_policy


if __name__ == "__main__":
    env = gymnasium.make('JacksCarRental-v0')
    R = env.reward
    P = env.transition
    print('Reward shape:', R.shape)
    print('Transition prob. shape:', P.shape)

    policy = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.int32)
    # values = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.float32)
    # P = P.transpose(1, 0, 2)

    policy = improve_policy(policy)
    ax = sns.heatmap(policy.reshape(21, 21))
    ax.invert_yaxis()
    plt.show()