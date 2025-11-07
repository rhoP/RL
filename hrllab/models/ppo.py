import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import scipy.signal
import numpy as np
from gymnasium.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]





class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf) 
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}




class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()

        self._obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]

        self._actor = nn.Sequential(
            nn.Linear(self._obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self._action_dim),
            nn.Softmax(dim=-1),
        )

        self._critic = nn.Sequential(
            nn.Linear(self._obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def pi(self, state, action=None):
        logits = self._actor(state)
        dist = Categorical(logits)
        action_logprobs = None
        if action is not None:
            action_logprobs = dist.log_prob(action)
        return dist, action_logprobs
    
    def v(self, state):
        return self._critic(state)

    def forward(self, state):
        # TODO: check if this should be with no_grad or not???
        dist, _ = self.pi(state)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)#TODO: This changes for normal, check spinning up
        dist_entropy = dist.entropy()
        state_vals = self._critic(state)
        return action.item(), action_logprobs.item(), state_vals.item(), dist_entropy.item()





class PPO:
    def __init__(self, env, lr_actor, lr_critic, gamma=0.99, clip_eps=0.2, n_epochs=1000):

        self._clip_eps = clip_eps
        self._gamma = gamma
        self._n_epochs = n_epochs

        self._ac = ActorCritic(env)
        self._old_ac = ActorCritic(env)
        self._old_ac.load_state_dict(self._ac.state_dict())

        self._optim_actor = AdamW(self._ac._actor.parameters(), lr=lr_actor)
        self._optim_critic = AdamW(self._ac._critic.parameters(), lr=lr_critic)

        self._loss_fn = nn.MSELoss()
        self._buffer = RolloutBuffer()


    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, logprob, val, entropy = self._old_ac(state)
        #self._buffer.update_buffer(state, action, logprob, val, entropy)
        return action # Tensor or what?

    def policy_loss(self, data):
        state, action, adv, logp_old = data['state'], data['act'], data['adv'], data['logp']

        pi, logp = self._ac.pi(state, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self._clip_eps, 1+self._clip_eps)*adv
        loss_pi = -(torch.min(ratio*adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self._clip_eps) | ratio.lt(1-self._clip_eps)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent,cf=clipfrac)

        return loss_pi, pi_info

    def value_loss(self, data):
        state, ret = data['state'], data['ret']
        return ((self._ac.v(state) - ret)**2).mean()

    def update(self):


