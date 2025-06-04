import torch
import torch.nn as nn
from torch.optim import AdamW


class RolloutBuffer:
    def __init__(self) -> None:
        self._actions = []
        self._states = []
        self._logprobs = []
        self._rewards = []
        self._state_vals = []
        self._dones = []

    def reset(self):
        del self._actions[:]
        del self._states[:]
        del self._logprobs[:]
        del self._rewards[:]
        del self._state_vals[:]
        del self._dones[:]


class ActorCriticPolicy(nn.Module):
    def __init__(self, env):
        super(ActorCriticPolicy, self).__init__()

        self._obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]

        self._actor = nn.Sequential(
            nn.Linear(self._obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self._action_dim),
            nn.Softmax(dim=-1),
        )

        self._critic = nn.Sequential(
            nn.Linear(self._obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def get_action_dist(self, state):
        actions_mu = self._actor(state)
        dist = torch.distributions.Categorical(actions_mu)
        return dist

    def get_q_value(self, state, action):
        dist = self.get_action_dist(state)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_vals = self._critic(state)
        return action_logprobs, state_vals, dist_entropy

    def forward(self):
        raise NotImplementedError


class PPO:
    def __init__(self, env, lr_actor, lr_critic, gamma, clip_eps, n_epochs):
        self._policy = ActorCriticPolicy(env)
        self._old_policy = ActorCriticPolicy(env)
        self._old_policy.load_state_dict(self._policy.state_dict())

        self._optim_actor = AdamW(self._policy._actor.parameters(), lr=lr_actor)
        self._optim_critic = AdamW(self._policy._critic.parameters(), lr=lr_critic)

        self._loss_fn = nn.MSELoss()
