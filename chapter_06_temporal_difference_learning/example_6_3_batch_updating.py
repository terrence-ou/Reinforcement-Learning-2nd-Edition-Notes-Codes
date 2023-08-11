import numpy as np
import matplotlib.pyplot as plt
from envs.random_walk_env import RandomWalk

THRESHOLD = 1e-3

########################################################################
###                          Helper Functions                        ###
########################################################################


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


def td_single_batch(
    V: np.ndarray, alpha: float, trajs: list, gamma: float = 1.0
) -> np.ndarray:
    while True:
        increments = np.zeros_like(V)
        for traj in trajs:
            for state, next_state, reward in traj:
                increments[state] += alpha * (reward + gamma * V[next_state] - V[state])
        if np.abs(increments).sum() < THRESHOLD:
            break
        V += increments
    return V


########################################################################
###                     MC Batch Update algorithm                    ###
########################################################################


def mc_single_batch(
    V: np.ndarray, alpha: float, trajs: list, gamma: float = 1.0
) -> np.ndarray:
    while True:
        increments = np.zeros_like(V)
        for traj in trajs:
            G = 0
            for i in range(len(traj) - 1, -1, -1):
                state, _, reward = traj[i]
                G = gamma * G + reward
                increments[state] += alpha * (G - V[state])
        if np.abs(increments).sum() < THRESHOLD:
            break
        V += increments
    return V


########################################################################
###                         Wraper functions                         ###
########################################################################


# evaluate the selected algorithm
def evaluation(num_episodes: int, alpha: float, mode: str) -> np.ndarray:
    assert mode in ["MC", "TD"], "mode should be either MC or TD"

    env = RandomWalk()
    total_states = len(env.state_space) + 2

    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] = 0.5
    V_history = np.zeros(shape=(num_episodes, total_states))
    batch = []  # a list to store all trajectories as batch

    for episode in range(num_episodes):
        traj = generate_trajectory(env)
        batch.append(traj)
        if mode == "MC":
            V = mc_single_batch(V, alpha, batch)
        else:
            V = td_single_batch(V, alpha, batch)
        V_history[episode] = np.copy(V)

    return V_history


# run the example
def run_bacth_update(num_episodes: int, num_runs: int, alpha: float) -> None:
    true_value = np.arange(0, 1.1, 1 / 6)[1:-1]
    for mode in ["TD", "MC"]:
        V_hist = np.zeros(shape=(num_runs, num_episodes, 5))
        for i in range(num_runs):
            error_hist = evaluation(num_episodes, alpha, mode=mode)
            V_hist[i] = error_hist[:, 1:-1]
        avg_error = rms(V_hist, true_value)
        plt.plot(avg_error, label=mode)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_episodes = 100
    num_runs = 1
    alpha = 1e-3
    run_bacth_update(num_episodes, num_runs, alpha)
