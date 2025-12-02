import numpy as np
import gymnasium as gn
import stable_baselines3 as sb3


'''
Here is the outline for the algorithm:
1. Identify the set B
2. Construct set A and setup the corresponding advantage estimators
3. Using the set A, steer the policy to the corresponding option
'''

