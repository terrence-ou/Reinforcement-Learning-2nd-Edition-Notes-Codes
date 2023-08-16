from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[-1])
    return np.argmax(Q[state], axis=-1)


def SARSA(env):
    raise NotImplementedError


def Q_learning(
    env: Any, epsilon: float, alpha: float, gamma: float, num_episodes: int = 500
) -> np.ndarray:
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))

    reward_hist = np.zeros(shape=(num_episodes,))

    for i in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        while True:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * Q[next_state].max() - Q[state, action]
            )
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        reward_hist[i] = episode_reward
    return reward_hist


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 1.0
    gamma = 1.0

    env = gym.make("CliffWalking-v0")

    reward_hist_total = np.zeros(shape=(100, 500))
    for i in range(100):
        reward_hist_total[i] = Q_learning(env, epsilon, alpha, gamma)

    plt.plot(reward_hist_total.mean(axis=0))
    plt.ylim(-100, 0)
    plt.show()
