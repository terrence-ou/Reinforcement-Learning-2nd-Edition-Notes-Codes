import numpy as np


def get_actions(stake:int):
    return np.arange(0, stake+1)



if __name__ == "__main__":
    V = np.zeros(shape=(100,))
    V[-1] = 1
    
    