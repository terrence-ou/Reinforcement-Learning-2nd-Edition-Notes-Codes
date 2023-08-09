import numpy as np
import matplotlib.pyplot as plt

# Import the RandomWalk environment
from envs.random_walk_env import RandomWalk


########################################################################
###                     TD and MC algorithms                         ###
########################################################################


# A random policy that return each action with equal probabilities
def random_policy() -> int:
    return np.random.choice(2)


# TD algorithm
def TD_evaluation(alpha: float = 0.1, num_episodes=100) -> np.ndarray:
    env = RandomWalk()
    total_states = len(env.state_space) + 2  # also include terminal states in both end
    # Initialize value
    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] += 0.5  # initial values of non-terminal state to 0.5
    gamma = 1.0

    V_history = np.zeros(shape=(num_episodes, total_states))

    # Run TD algorithm
    for episode in range(num_episodes):
        env.reset()
        state = env.initial_idx
        terminated = False
        while not terminated:
            action = random_policy()
            next_state, reward, terminated = env.step(action)
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
        V_history[episode] = np.copy(V)
    return V_history


# MC algorithm
def MC_evaluation(alpha: float = 0.1, num_episodes=100) -> np.ndarray:
    raise NotImplementedError


########################################################################
###                       Example wrappers                           ###
########################################################################


# For the value approximation example
def estimate_value(num_episodes: int, true_value: np.ndarray) -> None:
    V_hist_single = TD_evaluation(alpha=0.1, num_episodes=num_episodes)
    selected_hist = V_hist_single[(0, 9, 99), 1:-1]

    plt.plot(true_value)

    for hist in selected_hist:
        plt.plot(hist)
    plt.show()


# Rooted Mean Square Error
def rms(V_hist: np.ndarray, true_value: np.ndarray) -> np.ndarray:
    if len(true_value.shape) != 3:
        true_value = true_value.reshape(1, 1, -1)
    squared_error = (V_hist - true_value) ** 2
    rooted_mse = np.sqrt(squared_error.mean(axis=-1)).mean(axis=0)
    return rooted_mse


def parameter_sweep_td(
    alpha_list: list, num_episodes: int, true_value: np.ndarray
) -> None:
    num_runs = 100
    V_hist = np.zeros(shape=(num_runs, num_episodes, 5))
    for alpha in alpha_list:
        for i in range(num_runs):
            v_single = TD_evaluation(alpha=alpha)
            V_hist[i] = v_single[:, 1:-1]

        error = rms(V_hist, true_value)
        plt.plot(error)
    plt.show()


def parameter_sweep_mc(alpha_list, num_episodes, true_value) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    # Environment checking
    # check_RandomWalk_nodes()

    # The true value of the random policy for RW is provided as follows
    true_value = np.arange(0, 1.1, 1 / 6)[1:-1]
    num_episodes = 100
    alpha_list = [0.05, 0.1, 0.15]

    # Example a: value function approximation
    estimate_value(num_episodes=num_episodes, true_value=true_value)

    # Example b: RMS error over different setups
    # parameter_sweep_td(alpha_list, num_episodes, true_value)
