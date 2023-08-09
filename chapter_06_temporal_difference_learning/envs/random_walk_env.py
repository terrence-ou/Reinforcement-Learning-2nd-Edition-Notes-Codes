import numpy as np

########################################################################
###            Setting up the Random Walk environment                ###
########################################################################


# Using node to represent the states and the connection between each pair
class Node:
    def __init__(self, val: str):
        self.value = val
        self.right = None
        self.left = None
        self.r_reward = 0  # the reward of stepping right
        self.l_reward = 0  # the reward of stepping right

    def __eq__(self, other_val) -> bool:
        return self.value == other_val

    def __repr__(self) -> str:
        return f"Node {self.value}"


# Build the Random Walk environment
class RandomWalk:
    def __init__(self):
        self.state_space = ["A", "B", "C", "D", "E"]
        # We need to make the mapping start from 1 and reserve 0 for the terminal state
        self.state_idx_map = {
            letter: idx + 1 for idx, letter in enumerate(self.state_space)
        }
        self.initial_state = "C"
        self.initial_idx = self.state_idx_map[self.initial_state]
        # Build environment as a linked list
        self.nodes = self.build_env()
        self.reset()

    def step(self, action: int) -> tuple:
        assert action in [0, 1], "Action should be 0 or 1"

        if action == 0:
            reward = self.state.l_reward
            next_state = self.state_idx_map[self.state.value] - 1
            self.state = self.state.left
        else:
            reward = self.state.r_reward
            next_state = self.state_idx_map[self.state.value] + 1
            self.state = self.state.right

        terminated = False if self.state else True
        return next_state, reward, terminated

    # reset the state to the initial node
    def reset(self):
        self.state = self.nodes
        while self.state != self.initial_state:
            self.state = self.state.right

    # building the random walk environment as a linked list
    def build_env(self) -> Node:
        values = self.state_space
        head = Node(values[0])
        builder = head
        prev = None
        for i, val in enumerate(values):
            next_node = None if i == len(values) - 1 else Node(values[i + 1])
            if not next_node:
                builder.r_reward = 1

            builder.left = prev
            builder.right = next_node
            prev = builder
            builder = next_node
        return head


# Check the nodes and rewards
def check_RandomWalk_nodes():
    # Check environment setup
    print("\n=====Test 1, checking environment setup=====\n")
    head = RandomWalk().build_env()
    while head:
        print("Links: \t", head.left, "←", head, "→", head.right)
        print("Reward: \t", head.l_reward, "←", head, "→", head.r_reward, "\n")
        head = head.right

    # Check if the right-forward sequence correct
    print("\n=====Test 2: move all the way right=====\n")

    env = RandomWalk()
    terminated = False
    print("Initial state: ", env.state, env.initial_idx)
    while not terminated:
        next_state, reward, terminated = env.step(1)
        print(
            env.state, "\t", next_state, "reward: ", reward, "terminated:", terminated
        )

    # Check if the left-forward sequence correct
    print("\n=====Test 3: move all the way left=====\n")

    env.reset()
    terminated = False
    print("Initial state: ", env.state, env.initial_idx)
    while not terminated:
        next_state, reward, terminated = env.step(0)
        print(
            env.state, "\t", next_state, "reward: ", reward, "terminated:", terminated
        )

    # Check if random move generate correct trajectory
    print("\n=====Test 4: Random moves=====\n")
    env.reset()
    terminated = False
    print("Initial state: ", env.state, env.initial_idx)
    while not terminated:
        rand_action = np.random.choice(2)
        next_state, reward, terminated = env.step(rand_action)
        print(
            env.state, "\t", next_state, "reward: ", reward, "terminated:", terminated
        )

    print("\n=======Test Finished!=======\n")


if __name__ == "__main__":
    check_RandomWalk_nodes()
