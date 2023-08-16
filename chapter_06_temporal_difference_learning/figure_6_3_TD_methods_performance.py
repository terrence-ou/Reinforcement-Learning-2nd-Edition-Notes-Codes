from typing import Any
from collections import defaultdict
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


# Epsilon-greedy policy
def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float):
    if np.random.random() < epsilon:
        return np.random.choice(Q.shape[-1])
    return np.argmax(Q[state], axis=-1)


# Get the expected value of given state
def get_expectation(Q: np.ndarray, state: int, epsilon: float):
    # Non greedy actions
    result = (Q[state] * epsilon).sum()
    # Greedy action
    result += Q[state].max() * (1 - epsilon)
    return result


# General TD algorithm
def TD_algorithm(
    env: Any,
    update_rule: Any,
    epsilon: float,
    alpha: float,
    gamma: float,
    num_episodes: int,
) -> float:
    # initialize action-value function(Q table)
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros(shape=(nS, nA))

    reward_hist = np.zeros(shape=(num_episodes,))

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = update_rule(
            env=env, state=state, Q=Q, alpha=alpha, gamma=gamma, epsilon=epsilon
        )
        reward_hist[i] = episode_reward

    return reward_hist.mean()


########################################################################
###                             Algorithms                           ###
########################################################################


def SARSA(
    env: Any, state: int, Q: np.ndarray, alpha: float, gamma: float, epsilon: float
) -> float:
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
    return episode_reward


def Q_learning(
    env: Any, state: int, Q: np.ndarray, alpha: float, gamma: float, epsilon: float
) -> float:
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
    return episode_reward


def expected_SARSA(
    env: Any, state: int, Q: np.ndarray, alpha: float, gamma: float, epsilon: float
) -> float:
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
    return episode_reward


########################################################################
###                  TD comparison wrapper function                  ###
########################################################################


# Run a single algorithm
def run_algorithm(
    algorithm: Any, num_runs: int, num_episodes: int, results: dict, algo_key: str
) -> None:
    epsilon = 0.1
    gamma = 1.0
    alpha = np.arange(1, 11, 1, dtype=float) / 10

    env = gym.make("CliffWalking-v0")

    for a in alpha:
        running_reward = 0
        for _ in tqdm(range(num_runs), desc=f"{algo_key}, alpha={a}"):
            running_reward += TD_algorithm(
                env=env,
                update_rule=algorithm,
                epsilon=epsilon,
                alpha=a,
                gamma=gamma,
                num_episodes=num_episodes,
            )
        results[algo_key].append(running_reward / num_runs)


# Run all algorithms
def run_comparison(
    num_episodes: int, num_runs: int, prefix: str, results: dict
) -> None:
    algorithms = [expected_SARSA, Q_learning, SARSA]
    keys = ["expected_SARSA", "Q_learning", "SARSA"]
    for i, algorithm in enumerate(algorithms):
        run_algorithm(
            algorithm=algorithm,
            num_runs=num_runs,
            num_episodes=num_episodes,
            results=results,
            algo_key=prefix + keys[i],
        )


if __name__ == "__main__":
    num_episodes = [100_000, 100]
    num_runs = [10, 50_000]
    prefixs = ["Asymptotic ", "Interim "]
    train = True
    results = defaultdict(list)

    if train:
        for i in range(len(num_episodes)):
            episodes = num_episodes[i]
            runs = num_runs[i]
            prefix = prefixs[i]
            run_comparison(episodes, runs, prefix, results)
            with open("./history/figure_6_3/results.pkl", "wb") as f:
                pickle.dump(results, f)
    else:
        with open("./history/figure_6_3/results.pkl", "rb") as f:
            results = pickle.load(f)

        for key, value in results.items():
            plt.plot(value, label=key)
        plt.ylim(-150, 0)
        plt.legend()
        plt.show()
