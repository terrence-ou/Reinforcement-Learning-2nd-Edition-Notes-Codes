import numpy as np
from race_track_env.race_track import RaceTrack


if __name__ == "__main__":
    track = 'b'
    render_mode = None
    cell_size = 20
    env = RaceTrack(track, render_mode, cell_size)
    env.reset()
    terminated = False
    total_reward = 0
    while not terminated:
        action = np.random.choice(env.nA)
        observation, reward, terminated = env.step(action)
        total_reward += reward
    print(observation, reward, terminated, total_reward)