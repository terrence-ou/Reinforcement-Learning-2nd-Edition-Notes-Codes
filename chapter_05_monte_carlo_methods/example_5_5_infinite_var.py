import numpy as np

# Env for this example
def random_move(action):
    """
    actions: 0 -> go left; 1: -> go right
    states: 0 -> nonterminal state; 1 -> terminal state
    """
    if action == 0:
        prob = np.random.randn()
        if prob <= 0.1:
            return ()





if __name__ == "__main__":
    pass
    print(np.random.randn())