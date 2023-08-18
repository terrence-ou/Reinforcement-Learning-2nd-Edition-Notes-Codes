from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from envs.two_state_mdp import Two_State_MDP


# Epsilon-greedy policy
def epsilon_greedy(Q_state, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.choice(Q_state.shape[0])
    max_val = Q_state.max()
    max_acts = np.where(Q_state == max_val)[0]  # break the tie randomly
    return np.random.choice(max_acts)


# Q-learning algorithm
def Q_learning(
    env: Any,
    epsilon: float,
    alpha: float,
    gamma: float,
    left_actions: np.ndarray,
    num_episodes: int = 300,
) -> None:
    # Initialize action-value function (Q table)
    Q = [np.zeros(shape=n, dtype=float) for n in env.nS]

    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action = epsilon_greedy(Q[state], epsilon)

            if state == env.start_state:
                left_actions[i] += 1 - action

            next_state, reward, terminated = env.step(action)
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * Q[next_state].max() - Q[state][action]
            )
            state = next_state
            episode_reward += reward
            if terminated:
                break


# Double Q-learning algorithm
def double_Q_learning(
    env: Any,
    epsilon: float,
    alpha: float,
    gamma: float,
    left_actions: np.ndarray,
    num_episodes: int = 300,
) -> None:
    Q_1 = [np.zeros(shape=n, dtype=float) for n in env.nS]
    Q_2 = [np.zeros(shape=n, dtype=float) for n in env.nS]

    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            # policy based on the summation of two Qs
            action = epsilon_greedy(Q_1[state] + Q_2[state], epsilon)

            if state == env.start_state:
                left_actions[i] += 1 - action

            next_state, reward, terminated = env.step(action)
            # Double Q update
            if np.random.random() <= 0.5:
                A_star = Q_1[next_state].argmax()
                Q_1[state][action] = Q_1[state][action] + alpha * (
                    reward + gamma * Q_2[next_state][A_star] - Q_1[state][action]
                )
            else:
                A_star = Q_2[next_state].argmax()
                Q_2[state][action] = Q_2[state][action] + alpha * (
                    reward + gamma * Q_1[next_state][A_star] - Q_2[state][action]
                )

            state = next_state
            episode_reward += reward
            if terminated:
                break


def plot_result(left_actions_q, left_actions_dq):
    font_dict = {"fontsize": 11}
    plt.figure(figsize=(9, 6), dpi=150)
    plt.grid(c="lightgray")
    plt.margins(0.02)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
    plt.xlabel("Episodes", fontdict=font_dict)
    plt.ylabel("% of left actions from A", fontdict=font_dict)
    plt.yticks([0.05, 0.25, 0.5, 0.75, 1.0], ["5%", "25%", "50%", "75%", "100%"])
    plt.ylim(0.0, 1.0)
    plt.title(
        "Comparison of Q-learning and Double Q-learning",
        fontsize=13,
        fontweight="bold",
    )
    plt.plot(left_actions_q / num_runs, c="tomato", linewidth=1.5, label="Q-learning")
    plt.plot(
        left_actions_dq / num_runs,
        c="mediumseagreen",
        linewidth=1.5,
        label="Double Q-learning",
    )
    plt.plot(
        [0, 300],
        [0.05, 0.05],
        c="orchid",
        label="Optimal",
        linewidth=1.5,
        linestyle="dashed",
    )
    plt.legend(loc=7)
    plt.savefig("./plots/example_6_7.png")
    plt.show()


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.1
    gamma = 1
    num_episodes = 300
    num_runs = 10_000
    left_actions_q = np.zeros(shape=(num_episodes,))
    left_actions_dq = np.zeros(shape=(num_episodes,))

    env = Two_State_MDP()
    for _ in range(num_runs):
        Q_learning(env, epsilon, alpha, gamma, left_actions_q, num_episodes)
        double_Q_learning(env, epsilon, alpha, gamma, left_actions_dq, num_episodes)

    plot_result(left_actions_q, left_actions_dq)
