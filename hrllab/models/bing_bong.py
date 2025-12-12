########################################################################
#  Bing Bong algorithm currently with CO on I_0  #
########################################################################
import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union
import numpy as np
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict

class HierarchicalAlgorithm():
    """
    A general framework for a hierarchical actor
    """

    def __init__(self):
        self.nu = None  # state transformation
        self.U = None

    def setup_model(
        env,
    ):
        """
        Initialize nu and U
        Start with discrete state spaces and then move onto continous
        """

        # Check if the observation space is discrete
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError(
                f"Expected discrete observation space but received {type(env.observation_space).__name__} instead."
            )


        self.set_nu() 
        self.set_U() 

    def set_nu(m=3):
        """
        The state abstraction layer transforms observations into an ultrametric space 
        m: Defines the number of hierarchies
        """
        raise NotImplementedError

    def set_U():
        """
        The Unconscious layer 
        Produces a target set in the ultrametric space induced by 
        """
        raise NotImplementedError




class PolicyAlgorithm(BaseAlgorithm):
    """
    """
    def __init__():
        super().__init__():





























class BingBong(HierarchicalAlgorithm):
    """
    Custom algorithm that finds necessary behaviors defined as an interaction on the simplicial complex of states.

    Attributes that define the CO inner loop:

    k <- for the interaction level to search in, or the maximum interactions to search for
    inner_alg <- Which CO method to use
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




    
