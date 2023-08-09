import numpy as np


class Node:
    def __init__(self, val: str):
        self.value = val
        self.right = None
        self.left = None

    def __eq__(self, other_val) -> bool:
        return self.value == other_val


# Build the Random Walk environment
class RandomWalk:
    def __init__(self):
        self.nodes = self.build_env()

    def step(self):
        raise NotImplementedError

    # building the random walk environment as a linked list
    def build_env(self):
        values = ["A", "B", "C", "D", "E"]
        head = Node(values[0])
        builder = head
        prev = None
        for i, val in enumerate(values):
            next = None if i == len(values) - 1 else Node(values[i + 1])
            builder.left = prev
            builder.right = next
            prev = builder
            builder = next
        return head


# TODO: write a test block here
def test_RandomWalk_nodes():
    fake_env = RandomWalk()
    head = fake_env.nodes
    while head:
        print(head.value)
        head = head.right


if __name__ == "__main__":
    test_RandomWalk_nodes()
