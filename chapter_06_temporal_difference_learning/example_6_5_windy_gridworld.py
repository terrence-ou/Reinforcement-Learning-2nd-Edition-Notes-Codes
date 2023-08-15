from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from envs.windy_grid_env import WindyGridworld


# e-greedy policy
def epsilon_greedy(Q: np.ndarray, state: tuple, epsilon: float = 0.1) -> int:
    if np.random.random(size=(1,)) < epsilon:
        return np.random.choice(Q.shape[-1])
    return Q[state].argmax()


# SARSA Algorithm
def SARSA(env: Any, alpha: float, epsilon: float, total_steps: int = 8000) -> tuple:
    gamma = 1.0

    # Initialize Q values
    Q = np.zeros(shape=(*env.nS, env.nA))
    Q[env.goal] = 0  # Set the value of the terminal state to 0

    curr_step = 0
    total_episodes = 0
    step_episode_hist = np.zeros(shape=(total_steps + 1,), dtype=int)
    reward_hist = []

    # Run SARSA for given steps
    while curr_step < total_steps:
        episode_reward = 0
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        while True:
            curr_step += 1
            next_state, reward, terminated, truncated = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )
            state = next_state
            action = next_action
            episode_reward += reward
            step_episode_hist[curr_step] = total_episodes
            # stop conditions
            if terminated or truncated:
                total_episodes += 1
                reward_hist.append(episode_reward)
                break
            if curr_step >= total_steps:
                break

    return Q, step_episode_hist, reward_hist


# The wrapper function for windy gridworld example
def run_sarsa_windy(
    epsilon: float,
    alpha: float,
    total_steps: int,
    plot: bool = True,
    render_result: bool = False,
) -> None:
    env = WindyGridworld()
    Q, step_episode_hist, reward_hist = SARSA(
        env, alpha=alpha, epsilon=epsilon, total_steps=total_steps
    )

    # plot the result
    if plot:
        plt_setup(xlabel="Steps/Walks", ylabel="Episodes", title="Step-Episode history")
        plt.plot(step_episode_hist[:8000], linewidth=1.2, c="tomato")
        plt.savefig("./plots/example_6_5/step_episodes.png")
        plt.show()
        plt_setup(xlabel="Episodes", ylabel="Rewards", title="Reward history")
        plt.ylim((-300, 0))
        plt.plot(reward_hist[: step_episode_hist[8000]], linewidth=1.2, c="steelblue")
        plt.savefig("./plots/example_6_5/rewards.png")
        plt.show()

    # render out the animation of the learned optimal path
    if render_result:
        A = Q.argmax(axis=-1)
        env = WindyGridworld(render_mode="human")
        state = env.reset()
        while True:
            action = A[state]
            next_state, reward, terminated, truncated = env.step(action)
            state = next_state
            if terminated or truncated:
                break


# Setup plot visualization basis
def plt_setup(xlabel: str, ylabel: str, title: str = None):
    # codes for plotting
    font_dict = {"fontsize": 11}
    plt.figure(figsize=(6, 6), dpi=150)
    plt.grid(c="lightgray")
    plt.margins(0.02)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
    plt.xlabel(xlabel, fontdict=font_dict)
    plt.ylabel(ylabel, fontdict=font_dict)
    if title:
        plt.title(title, fontsize=13, fontweight="bold")


if __name__ == "__main__":
    epsilon = 0.1
    alpha = 0.5
    total_steps = 20_000
    run_sarsa_windy(
        epsilon=epsilon,
        alpha=alpha,
        total_steps=total_steps,
        plot=True,
        render_result=False,
    )
