import numpy as np
import matplotlib.pyplot as plt
from envs.random_walk_env import RandomWalk


# Random policy
def random_policy() -> int:
    return np.random.choice(2)


# Rooted Mean Square Error
def rms(V_hist: np.ndarray, true_value: np.ndarray) -> np.ndarray:
    if len(true_value.shape) != 3:
        true_value = true_value.reshape(1, 1, -1)
    squared_error = (V_hist - true_value) ** 2
    rooted_mse = np.sqrt(squared_error.mean(axis=-1)).mean(axis=0)
    return rooted_mse


# Generate a trajectory based on the environment and random policy
def generate_trajectory(env: RandomWalk) -> list:
    trajectory = []
    env.reset()
    state = env.initial_idx
    terminated = False
    while not terminated:
        action = random_policy()
        next_state, reward, terminated = env.step(action)
        trajectory.append((state, next_state, reward))
        state = next_state
    return trajectory


########################################################################
###                     TD Batch Update algorithm                    ###
########################################################################


# update by visiting all trajectories in the batch
def td_single_batch(
    V: np.ndarray, alpha: float, trajs: list, gamma: float = 1.0
) -> np.ndarray:
    increments = np.zeros_like(V)
    for traj in trajs:
        for state, next_state, reward in traj:
            increments[state] += alpha * (reward + gamma * V[next_state] - V[state])

    V += increments
    return V


# TD evaluation algorithm
def td_evaluation(num_episodes: int, alpha: float):
    env = RandomWalk()
    total_states = len(env.state_space) + 2

    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] = 0.5
    V_history = np.zeros(shape=(num_episodes, total_states))
    batch = []  # a list to store all trajectories as batch

    for episode in range(num_episodes):
        traj = generate_trajectory(env)
        batch.append(traj)
        V = td_single_batch(V, alpha, batch)
        V_history[episode] = np.copy(V)

    return V_history


########################################################################
###                     MC Batch Update algorithm                    ###
########################################################################
def mc_batch_update():
    raise NotImplementedError


def mc_updating():
    raise NotImplementedError


########################################################################
###                         Wraper functions                         ###
########################################################################


if __name__ == "__main__":
    num_episodes = 100
    num_runs = 100
    true_value = np.arange(0, 1.1, 1 / 6)[1:-1]
    alpha = 4e-3
    V_hist = np.zeros(shape=(num_runs, num_episodes, 5))
    for i in range(num_runs):
        td_error_hist = td_evaluation(num_episodes, alpha)
        V_hist[i] = td_error_hist[:, 1:-1]
    error = rms(V_hist, true_value)
    plt.plot(error)
    plt.show()
