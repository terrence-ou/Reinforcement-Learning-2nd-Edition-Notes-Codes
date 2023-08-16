from typing import Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


# Epsilon-greedy policy
def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[-1])
    return np.argmax(Q[state], axis=-1)


########################################################################
###                    Expected SARSA algorithms                     ###
########################################################################


def get_expectation(Q: np.ndarray, state: int, epsilon: float):
    # Non greedy actions
    result = (Q[state] * epsilon).sum()
    # Greedy action
    result += Q[state].max() * (1 - epsilon)
    return result


def expected_SARSA(
    env: Any, epsilon: float, alpha: float, gamma: float, num_episodes: int = 100_000
) -> None:
    # initialize action-value function (Q table)
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
            expect_val = get_expectation(Q, next_state, epsilon)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * expect_val - Q[state, action]
            )
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        reward_hist[i] = episode_reward
    return reward_hist.mean()


########################################################################
###                  TD comparison wrapper function                  ###
########################################################################


def run_td_comparison():
    raise NotImplementedError


if __name__ == "__main__":
    epsilon = 0.1
    gamma = 1.0
    alpha = np.arange(1, 11, 1, dtype=float) / 10
    num_episodes = 100_00

    result = defaultdict(list)

    env = gym.make("CliffWalking-v0")
    for a in alpha:
        avg_reward = expected_SARSA(
            env=env, epsilon=epsilon, alpha=a, gamma=gamma, num_episodes=num_episodes
        )
        result["expected_SARSA"].append(avg_reward)

    plt.plot(result["expected_SARSA"])
    plt.ylim(-150, 0)
    plt.show()
