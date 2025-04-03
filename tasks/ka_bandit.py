import numpy as np
import random


np.random.seed(seed=42)


class K_armed_bandit():
    '''
    Instantiates a data class for the K-armed bandit task.
    

    The mean reward for each of the K arms is drawn from a 
    normal distribution with mean 0 and std 1.0.
    For the kth arm, a call to the _______ function instantiates a new normal 
    distribution with mean self._q_star[k] and std 1.0. 

    '''
    def __init__(self, K):
        self._K = K
        self._q_star = np.random.normal(loc=0.0, scale=1.0, size=self._K)

    # May need to change the name of this, perhaps a dunder method?
    def query(self, k):
        return np.random.normal(loc=self._q_star[k], scale=1.0, size=1)


    
