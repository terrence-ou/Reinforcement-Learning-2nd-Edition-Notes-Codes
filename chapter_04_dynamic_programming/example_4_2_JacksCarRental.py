import gym_jcr
import gymnasium
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

MAX_CAPACITY = 20
TRANSFER_MAX = 5
GAMMA = 0.9

# Plot values in 3D space
def plot_3d_value(values):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    values = values.reshape(21, 21)
    X = np.arange(0, MAX_CAPACITY + 1)
    Y = np.arange(0, MAX_CAPACITY + 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_wireframe(X, Y, values, rstride=1, cstride=1)
    plt.show()


def plot_policy(policy, iter=0):
    ax = sns.heatmap(policy.reshape(21, 21),
                     cmap='mako')
    ax.invert_yaxis()
    ax.set_xticks([0, 21], [0, 20])
    ax.set_yticks([0, 21], [0, 20])
    ax.set_xlabel('# Cars at second location')
    ax.set_ylabel('# Cars at first location')
    ax.set_title(f'$\pi_{iter}$', fontsize=16)
    plt.show()



if __name__ == "__main__":
    env = gymnasium.make('JacksCarRental-v0')
    R = env.reward
    P = env.transition
    print('Reward shape:', R.shape)
    print('Transition prob. shape:', P.shape)

    # 1. Initialization
    policy = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.int32)
    values = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.float32)
    converged = False


    while not converged:
        test_round = 0
        # 2. Policy evaluation
        while True:
            delta = 0
            old_values = np.copy(values)
            for s in range(env.nS):
                probs = P[:, s, policy[s] + TRANSFER_MAX]
                reward = R[s, policy[s] + TRANSFER_MAX]
                new_val = probs * (reward + GAMMA * values)
                values[s] = new_val.sum()
            diff = abs(old_values - values).max()
            delta = max(delta, diff)
            if delta < 1e-10: 
                break
        print(values.mean())

        #3. Policy improvement
        policy_stable = True
        old_policy = np.copy(policy)
        
        for s in range(env.nS):
            probs = P[:, s].T
            reward = R[s]
            pi_candidates = reward + GAMMA * probs @ values
            policy[s] = pi_candidates.argmax() - TRANSFER_MAX
            
            if policy[s] != old_policy[s]:
                policy_stable = False

        if policy_stable: break


    # plot_policy(policy, 4)
    plot_3d_value(values)