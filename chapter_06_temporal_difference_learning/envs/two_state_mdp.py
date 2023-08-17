import numpy as np


# Environment for example 6.7
class Two_State_MDP:
    def __init__(self):
        self.nS = [(1,), (20,), (2,), (1,)]
        self.nA = [1, 20, 2, 1]
        self.rewards = np.random.normal(loc=-0.1, scale=1.0, size=(20,))
        self.reset()

    def step(self, action):
        assert 0 <= action < self.nA[self.state], "Wrong action"
        terminated = False
        if self.state == 2:
            if action == 0:
                self.state -= 1
                reward = 0
            else:
                self.state += 1
                reward = 0
                terminated = True
        else:
            reward = self.rewards[action]
            self.state -= 1
            terminated = True
        return self.state, reward, terminated

    def reset(self):
        self.state = 2
        return self.state


# Test
if __name__ == "__main__":
    env = Two_State_MDP()

    for i in range(10):
        print("Episode: ", i)
        state = env.reset()
        while True:
            action = np.random.choice(env.nA[state])
            next_state, reward, terminated = env.step(action)
            print(state, next_state, reward, terminated)
            state = next_state
            if terminated:
                break
