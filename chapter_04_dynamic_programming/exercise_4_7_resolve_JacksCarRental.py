import gym_jcr
import gymnasium
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

MAX_CAPACITY = 20
TRANSFER_MAX = 5
GAMMA = 0.9


# Plot values in 3D space
def plot_3d_value(values, curr_iter, save=False):

    fig = plt.figure(figsize=plt.figaspect(0.8), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    values = values.reshape(21, 21)
    X = np.arange(0, MAX_CAPACITY + 1)
    Y = np.arange(0, MAX_CAPACITY + 1)
    X, Y = np.meshgrid(X, Y)

    ax.plot_wireframe(X, Y, values, rstride=1, cstride=1, linewidth=1.0, color='steelblue')

    ax.set_xticks(np.arange(0, 21, 5), [0] + [''] * 3 + [20])
    ax.set_yticks(np.arange(0, 21, 5), [0] + [''] * 3 + [20])
    ax.set_xlabel('# Cars at second location')
    ax.set_ylabel('# Cars at first location')

    ax.set_title(f"$v_{{\pi_{curr_iter}}}$", fontsize=16)
    if not save:
        plt.show()
    else:
        plt.savefig(f'./plots/exercise_4_7/value_{curr_iter}.png')


# Plot policy as a heatmap
def plot_policy(policy, curr_iter, save=False):
    plt.figure(figsize=plt.figaspect(0.8), dpi=300)
    ax = sns.heatmap(policy.reshape(21, 21),
                     cmap='mako')

    ax.invert_yaxis()
    ax.set_xticks([0, 21], [0, 20])
    ax.set_yticks([0, 21], [0, 20])
    ax.set_xlabel('# Cars at second location')
    ax.set_ylabel('# Cars at first location')
    ax.set_title(f'$\pi_{curr_iter}$', fontsize=16)

    if not save:
        plt.show()
    else:
        plt.savefig(f'./plots/exercise_4_7/policy_{curr_iter}.png')



if __name__ == "__main__":
    # Turn on the 'resolve' to taking non-linear rewards into training 
    env = gymnasium.make('JacksCarRental-v0', resolve=True)
    R = env.reward # The reward matrix
    P = env.transition # The transition probability matrix
    print('Reward shape:', R.shape)
    print('Transition prob. shape:', P.shape)

    # 1. Initialization
    policy = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.int32)
    values = np.zeros(shape=((MAX_CAPACITY+1)**2), dtype=np.float32)
    converged = False

    curr_iter = 1
    plot_policy(policy, 0, True)

    while not converged:
        # 2. Policy evaluation
        while True:
            delta = 0
            old_values = np.copy(values) # Backup old value
            for s in range(env.nS):
                # get transition probability based on current state and policy
                probs = P[:, s, policy[s] + TRANSFER_MAX]
                reward = R[s, policy[s] + TRANSFER_MAX]
                # calculate new value for the state
                new_val = probs * (reward + GAMMA * values)
                values[s] = new_val.sum()
            # che difference in values between old/new values
            diff = abs(old_values - values).max()
            delta = max(delta, diff)
            if delta < 1e-10: 
                break

        print("Current mean value: ", values.mean())

        #3. Policy improvement
        policy_stable = True
        old_policy = np.copy(policy) # Backup old policy
        
        for s in range(env.nS):
            # Get probability and reward matrix based on current state
            probs = P[:, s].T
            reward = R[s]
            # Get new policy
            pi_candidates = reward + GAMMA * probs @ values
            policy[s] = pi_candidates.argmax() - TRANSFER_MAX
            if policy[s] != old_policy[s]:
                policy_stable = False
            
        plot_policy(policy, curr_iter, True)
        if policy_stable: break
        curr_iter += 1

    plot_3d_value(values, curr_iter, True)