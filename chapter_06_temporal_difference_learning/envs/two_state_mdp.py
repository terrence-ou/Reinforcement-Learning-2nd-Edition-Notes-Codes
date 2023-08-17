import numpy as np


# Environment for example 6.7
class Two_State_MDP:
    def __init__(self):
        num_samples = 10
        self.nS = [(num_samples,), (2,), (1,)]
        self.nA = [num_samples, 2, 1]
        self.start_state = 1
        self.reset()

    def step(self, action):
        assert 0 <= action < self.nA[self.state], "Wrong action"
        terminated = False
        if self.state == self.start_state:
            if action == 0:
                self.state -= 1
                reward = 0
            else:
                self.state += 1
                reward = 0
                terminated = True
        else:
            reward = np.random.normal(-0.1, 1.0)
            self.state = 2
            terminated = True
        return self.state, reward, terminated

    def reset(self):
        self.state = 1
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
