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
    total_states = len(env.state_space) + 2  # also include terminal states on both ends
    # Initialize values
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
    env = RandomWalk()
    total_states = len(env.state_space) + 2  # include terminal states on both ends
    # Initialize values
    V = np.zeros(shape=(total_states), dtype=float)
    V[1:-1] += 0.5
    gamma = 1.0

    V_history = np.zeros(shape=(num_episodes, total_states))

    # Run MC algorithm
    for episode in range(num_episodes):
        env.reset()
        state = env.initial_idx
        terminated = False
        traj = []  # A list to store trajectories of the episode
        while not terminated:
            action = random_policy()
            next_state, reward, terminated = env.step(action)
            traj.append((state, action, reward))
            state = next_state

        G = 0  # The returns of the trajectory
        while traj:
            (state, action, reward) = traj.pop()
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])
        V_history[episode] = np.copy(V)
    return V_history


########################################################################
###                       Example wrappers                           ###
########################################################################


# For the value approximation example
def estimate_value(num_episodes: int, true_value: np.ndarray) -> None:
    V_hist_single = TD_evaluation(alpha=0.1, num_episodes=num_episodes)
    sel_epochs = (0, 9, 99)
    selected_hist = V_hist_single[sel_epochs, 1:-1]

    # Plotting the result
    font_dict = {"fontsize": 11}
    x_axis = np.arange(5, dtype=int)
    colors = ["mediumseagreen", "steelblue", "orchid"]

    plt.figure(figsize=(6, 6), dpi=150)
    plt.grid(c="lightgray")
    plt.margins(0.02)
    plt.xticks(x_axis, ["A", "B", "C", "D", "E"])
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
    plt.xlabel("State", fontdict=font_dict)
    plt.ylabel("Estimated value", fontdict=font_dict)

    plt.scatter(x_axis, true_value, c="black", s=10.0)
    plt.plot(x_axis, true_value, c="black", linewidth=1.6, label="True values")

    for i, hist in enumerate(selected_hist):
        plt.scatter(x_axis, hist, s=7.0, c=colors[i])
        plt.plot(
            x_axis, hist, linewidth=1.2, c=colors[i], label=f"sweep {sel_epochs[i]+1}"
        )
    plt.title("The values learned after $n$ sweeps", fontweight="bold", fontsize=13)
    plt.legend(loc=4)
    plt.savefig("./plots/example_6_2/value_approx.png")
    plt.show()


# Rooted Mean Square Error
def rms(V_hist: np.ndarray, true_value: np.ndarray) -> np.ndarray:
    if len(true_value.shape) != 3:
        true_value = true_value.reshape(1, 1, -1)
    squared_error = (V_hist - true_value) ** 2
    rooted_mse = np.sqrt(squared_error.mean(axis=-1)).mean(axis=0)
    return rooted_mse


# Run TD evaluation for given episodes
def parameter_sweep_td(
    alpha_list: list, num_episodes: int, num_runs: int, true_value: np.ndarray
) -> list:
    error_hist = []
    V_hist = np.zeros(shape=(num_runs, num_episodes, 5))
    for alpha in alpha_list:
        for i in range(num_runs):
            v_single = TD_evaluation(alpha=alpha, num_episodes=num_episodes)
            V_hist[i] = v_single[:, 1:-1]

        error = rms(V_hist, true_value)
        error_hist.append(error)
    return error_hist


# Run MC evaluation for given episodes
def parameter_sweep_mc(
    alpha_list: list, num_episodes: int, num_runs: int, true_value: np.ndarray
) -> list:
    error_hist = []
    V_hist = np.zeros(shape=(num_runs, num_episodes, 5))
    for alpha in alpha_list:
        for i in range(num_runs):
            v_single = MC_evaluation(alpha=alpha, num_episodes=num_episodes)
            V_hist[i] = v_single[:, 1:-1]

        error = rms(V_hist, true_value)
        error_hist.append(error)
    return error_hist


def algorithm_comparison(
    num_episodes: int, num_runs: int, true_value: np.ndarray
) -> None:
    # param sweeps on both algorithms
    alpha_list_mc = [0.01, 0.02, 0.03, 0.04]
    mc_error_hist = parameter_sweep_mc(
        alpha_list_mc, num_episodes, num_runs, true_value
    )
    alpha_list_td = [0.05, 0.1, 0.15]
    td_error_hist = parameter_sweep_td(
        alpha_list_td, num_episodes, num_runs, true_value
    )

    # Plotting the result
    font_dict = {"fontsize": 11}
    colors = ["mediumseagreen", "steelblue", "orchid"]

    plt.figure(figsize=(9, 6), dpi=150)
    plt.grid(c="lightgray")
    plt.margins(0.02)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)

    plt.xlabel("Walks/Episodes", fontdict=font_dict)
    plt.ylabel("RMS error", fontdict=font_dict)

    # plot MC errors with different line styles
    color = colors[2]
    line_types = [(5, (10, 3)), (0, (5, 5)), (0, (5, 0)), (0, (1, 1))]
    for i, error in enumerate(mc_error_hist):
        plt.plot(
            error,
            c=color,
            linewidth=1.5,
            linestyle=line_types[i],
            label=f"MC $\\alpha={alpha_list_mc[i]}$",
        )

    # plot TD errors with different line weights
    color = colors[1]
    linewidths = [1.8, 1.3, 0.9]
    for i, error in enumerate(td_error_hist):
        plt.plot(
            error,
            c=color,
            linewidth=linewidths[i],
            label=f"TD $\\alpha$={alpha_list_td[i]}",
        )

    plt.title(
        "Empirical RMS error, averaged over states and 100 runs",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend()
    plt.savefig("./plots/example_6_2/rms_compare.png")
    plt.show()


if __name__ == "__main__":
    # The true value of the random policy for RW is provided as follows
    true_value = np.arange(0, 1.1, 1 / 6)[1:-1]
    num_runs = 100
    num_episodes = 150

    # Example a: value function approximation
    estimate_value(num_episodes=num_episodes, true_value=true_value)

    # Example b: RMS error over different setups
    algorithm_comparison(num_episodes, num_runs, true_value)
