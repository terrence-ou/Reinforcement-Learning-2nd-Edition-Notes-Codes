import numpy as np
from envs.random_walk_env import RandomWalk


def random_policy() -> int:
    return np.random.choice(2)


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


def td_batch_update(
    V: np.ndarray, alpha: float, trajs: list, gamma: float = 1.0
) -> np.ndarray:
    increments = np.zeros_like(V)
    for traj in trajs:
        for state, next_state, reward in traj:
            increments[state] = alpha * (reward + gamma * V[next_state] - V[state])
    V += increments
    return V


def mc_batch_update():
    raise NotImplementedError


def td_evaluation(num_episodes: int):
    env = RandomWalk()
    total_states = len(env.state_space) + 2

    V = np.zeros(shape=(total_states), dtype=float)


def mc_updating():
    raise NotImplementedError


if __name__ == "__main__":
    env = RandomWalk()
    test_traj = generate_trajectory(env)
    print(test_traj)
