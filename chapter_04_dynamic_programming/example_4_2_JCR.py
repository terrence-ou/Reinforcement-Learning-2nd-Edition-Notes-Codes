import gym_jcr
import gymnasium
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

MAX_CAPACITY = 20
TRANSFER_MAX = 5
GAMMA = 0.9

def get_transition_kernel_for_policy(policy, P):
    return P[:, range((MAX_CAPACITY + 1) ** 2), policy + TRANSFER_MAX]

def get_P_reward_for_policy(policy, R, P):
    P_pi = get_transition_kernel_for_policy(policy, P)
    return P_pi, R[range((MAX_CAPACITY + 1) ** 2), policy + TRANSFER_MAX]

# Policy evaluation
def evaluate_policy(policy, R, P, values=np.zeros((MAX_CAPACITY + 1) ** 2)):
    P_pi, reward = get_P_reward_for_policy(policy, R, P)
    converged = False
    while not converged:
        new_values = reward + GAMMA * np.dot(P_pi.T, values)
        if np.allclose(new_values, values, rtol=1e-07, atol=1e-10):
            converged = True
        values = new_values
    return values


# Greedy policy
def greedy_improve(values, R, P):
    P_ = P.transpose(1, 0, 2)
    all_pure_states = np.eye((MAX_CAPACITY + 1) ** 2)
    new_states = np.dot(all_pure_states, P_)
    q = np.dot(new_states.transpose(2, 1, 0), values)
    q = q.T + R
    policy_indices = np.argmax(q, axis=1)
    policy = policy_indices - TRANSFER_MAX
    return policy

# Policy improvement
def improve_policy(policy, R, P):
    values = evaluate_policy(policy, R, P)
    not_converged = True
    while not_converged:
        print(values.mean())
        new_policy = greedy_improve(values, R, P)
        new_values = evaluate_policy(new_policy, R, P, values)
        if np.allclose(new_values, values, rtol=1e-02):
            not_converged = False
        values = new_values
    return new_policy




def plot3d_over_states(f, zlabel="", ):
    A = np.arange(0, MAX_CAPACITY+1)
    B = np.arange(0, MAX_CAPACITY+1)
    # B, A !!!
    B, A = np.meshgrid(B, A)
    V = f.reshape(MAX_CAPACITY+1,-1)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(A, B, V, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.scatter(A, B, V, c='b', marker='.')
    ax.set_xlabel("cars at A")
    ax.set_ylabel("cars at B")
    ax.set_zlabel(zlabel)
    
    ax.set_xticks(np.arange(0,21,1))
    ax.set_yticks(np.arange(0,21,1))
    #plt.xticks(np.arange(0,1,21))
    #ax.view_init(elev=10., azim=10)
    
    plt.show()

def plot_policy(policy):
    A = np.arange(0, MAX_CAPACITY+1)
    B = np.arange(0, MAX_CAPACITY+1)
    A, B = np.meshgrid(A, B)
    Po = policy.reshape(MAX_CAPACITY+1,-1)
    levels = range(-5,6,1)
    plt.figure(figsize=(7,6))
    CS = plt.contourf(A, B, Po, levels)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('actions')
    #plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Policy')
    plt.xlabel("cars at B")
    plt.ylabel("cars at A")
    plt.show()

if __name__ == "__main__":
    env = gymnasium.make('JacksCarRental-v0')
    R = env.reward
    P = env.transition
    print('Reward shape:', R.shape)
    print('Transition prob. shape:', P.shape)

    
    policy = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.int32)
    values = np.zeros(shape=((MAX_CAPACITY+1) ** 2), dtype=np.float32)
    # P = P.transpose(2, 0, 1)

    # values_ = evaluate_policy(policy, R, P, values)
    policy = improve_policy(policy, R, P)
    
    plot_policy(policy)