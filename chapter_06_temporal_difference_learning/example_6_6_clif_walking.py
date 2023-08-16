from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import gymnasium as gym


# Epsilon-greedy policy
def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[-1])
    return np.argmax(Q[state], axis=-1)


# SARSA algorithm
def SARSA(
    env: Any, epsilon: float, alpha: float, gamma: float, num_episodes: int = 500
) -> tuple:
    # initialize action-value function(Q table)
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))

    reward_hist = np.zeros(shape=(num_episodes,))

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        action = epsilon_greedy(Q, state, epsilon)
        # Single-episode pass
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )
            state = next_state
            action = next_action
            episode_reward += reward
            if terminated or truncated:
                break

        reward_hist[i] = episode_reward
    return Q, reward_hist


# Q-learning algorithm
def Q_learning(
    env: Any, epsilon: float, alpha: float, gamma: float, num_episodes: int = 500
) -> tuple:
    # Initialize action-value function (Q table)
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))

    reward_hist = np.zeros(shape=(num_episodes,))

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        # Single-episode pass
        while True:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state].max() - Q[state, action]
            )
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        reward_hist[i] = episode_reward
    return Q, reward_hist


# The wrapper function for
def run_q_sarsa_comparison(
    env: Any,
    total_runs: int,
    num_episodes: int,
    epsilon: float,
    alpha: float,
    gamma: float,
) -> tuple:
    reward_hist_Q = np.zeros(shape=(total_runs, 500))
    reward_hist_SAR = np.zeros(shape=(total_runs, 500))

    for i in range(total_runs):
        Q_Q, reward_hist_Q[i] = Q_learning(env, epsilon, alpha, gamma, num_episodes)
        Q_SARSA, reward_hist_SAR[i] = SARSA(env, epsilon, alpha, gamma, num_episodes)

    return (reward_hist_Q, reward_hist_SAR), (Q_Q, Q_SARSA)


# plot reward history
def plot_history(q_history, sarsa_history):
    # plot visualization setup
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
    plt.ylabel("Sum of rewards during each episode", fontdict=font_dict)
    plt.title(
        "Comparison between Q-learning and SARSA on Cliff Walking",
        fontsize=13,
        fontweight="bold",
    )
    # plot result
    q_history = uniform_filter(q_history.mean(axis=0), size=7)
    sarsa_history = uniform_filter(sarsa_history.mean(axis=0), size=7)
    plt.plot(sarsa_history, c="steelblue", label="SARSA")
    plt.plot(q_history, c="tomato", label="Q-learning")
    plt.ylim(-100, -15)
    plt.legend(loc=4)
    plt.savefig("./plots/example_6_6/rewards.png")
    plt.show()


# render out gameplay with given Q value
def render_result(Q):
    env = gym.make("CliffWalking-v0", render_mode="human")
    for _ in range(10):
        state, _ = env.reset()
        while True:
            action = Q[state].argmax()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break


if __name__ == "__main__":
    # Hyperparameters
    epsilon = 0.1
    alpha = 0.5
    gamma = 1.0

    total_runs = 80
    num_episodes = 500

    # Create environment
    env = gym.make("CliffWalking-v0")

    # Run both algorithm
    (q_reward, sarsa_reward), (Q_q, Q_SARSA) = run_q_sarsa_comparison(
        env=env,
        total_runs=total_runs,
        num_episodes=num_episodes,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
    )

    # plot result
    plot_history(q_history=q_reward, sarsa_history=sarsa_reward)
