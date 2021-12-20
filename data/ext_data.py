import numpy as np 

def dataset():
    with open('data/ds-ext.npy', 'rb') as f:
        x = np.load(f)
        y = np.load(f)
    def result(set):
        return (x, y)
    return result 
