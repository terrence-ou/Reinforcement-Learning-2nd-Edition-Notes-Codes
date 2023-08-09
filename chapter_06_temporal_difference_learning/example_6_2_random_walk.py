import numpy as np
import matplotlib.pyplot as plt


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
        self.states = ["A", "B", "C", "D", "E"]
        self.state_idx_map = {letter: idx for idx, letter in enumerate(self.states)}
        self.nodes = self.build_env()

    def step(self, action: int) -> tuple:
        raise NotImplementedError

    # building the random walk environment as a linked list
    def build_env(self) -> Node:
        values = self.states
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
    head = RandomWalk().build_env()
    while head:
        print("Links: \t", head.left, "←", head, "→", head.right)
        print("Reward: \t", head.l_reward, "←", head, "→", head.r_reward, "\n")
        head = head.right


if __name__ == "__main__":
    check_RandomWalk_nodes()
