########################################################################
#  Implementation of the bing bong algorithm currently with CO on I_0  #
########################################################################
import warnings
from typing import Any, ClassVar, Optional,TypeVar, Union

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import BaseAlgorithm




class BingBong(BaseAlgorithm):
        '''
        Custom algorithm that finds necessary behaviors defined as an interaction on the simplicial complex of states. 

        Attributes that define the CO inner loop:
        
        k <- for the interaction level to search in, or the maximum interactions to search for
        inner_alg <- Which CO method to use

        '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
