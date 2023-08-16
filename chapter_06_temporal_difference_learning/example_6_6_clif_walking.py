from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import gymnasium as gym


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[-1])
    return np.argmax(Q[state], axis=-1)


def SARSA(
    env: Any, epsilon: float, alpha: float, gamma: float, num_episodes: int = 500
) -> tuple:
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))

    reward_hist = np.zeros(shape=(num_episodes,))

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        action = epsilon_greedy(Q, state, epsilon)
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


def Q_learning(
    env: Any, epsilon: float, alpha: float, gamma: float, num_episodes: int = 500
) -> tuple:
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))

    reward_hist = np.zeros(shape=(num_episodes,))

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
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


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.5
    gamma = 1.0

    total_runs = 10
    total_episodes = 500

    env = gym.make("CliffWalking-v0")

    reward_hist_Q = np.zeros(shape=(total_runs, 500))
    reward_hist_SAR = np.zeros(shape=(total_runs, 500))

    for i in range(total_runs):
        Q_Q, reward_hist_Q[i] = Q_learning(env, epsilon, alpha, gamma)
        Q_SARSA, reward_hist_SAR[i] = SARSA(env, epsilon, alpha, gamma)

    plt.plot(uniform_filter(reward_hist_Q.mean(axis=0), size=7), label="Q-learning")
    plt.plot(uniform_filter(reward_hist_SAR.mean(axis=0), size=7), label="SARSA")
    plt.ylim(-100, 0)
    plt.legend()
    plt.show()

    env = gym.make("CliffWalking-v0", render_mode="human")
    state, _ = env.reset()
    while True:
        action = Q_SARSA[state].argmax()
        state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
