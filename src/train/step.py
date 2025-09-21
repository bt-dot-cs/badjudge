import numpy as np

def loss_fn(P:np.array,Q:np.array):
    return -np.sum(P * np.log(Q))
    
def step():
    pass