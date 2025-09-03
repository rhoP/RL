from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from hrllab.utils.custom_wrappers import CustomRewardWrapper


class PolicyTrackingWithBackwardGraphCallback(BaseCallback):
    """Callback to track policies and build backward graph during training."""

    def __init__(self, visualizer, backward_builder, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.visualizer = visualizer
        self.backward_builder = backward_builder
        self.save_freq = save_freq
        self.current_episode_states = []
        self.current_episode_actions = []

    def _on_step(self) -> bool:
        # Get current environment info (this will vary based on your env setup)
        # For VecEnv, we need to access the environments
        if hasattr(self.model.env, 'envs'):
            # For vectorized environments
            for env_idx, env in enumerate(self.model.env.envs):
                if hasattr(env, 'get_attr'):
                    # Try to get current state from environment
                    try:
                        state = env.get_attr('state', [0])[0]
                        action = self.model.env.get_attr('last_action', [0])[0] if hasattr(self.model.env,
                                                                                           'get_attr') else 0
                        reward = self.model.env.get_attr('last_reward', [0])[0] if hasattr(self.model.env,
                                                                                           'get_attr') else 0
                        done = self.model.env.get_attr('done', [0])[0] if hasattr(self.model.env, 'get_attr') else False

                        self.backward_builder.record_step(state, action, reward, state, done)
                    except:
                        pass

        if self.n_calls % self.save_freq == 0:
            # Save current policy
            policy_state = {
                'policy': self.model.policy.state_dict(),
                'step': self.n_calls
            }
            trajectory = self.visualizer.get_policy_trajectory(self.model)
            self.visualizer.policies.append(policy_state)
            self.visualizer.trajectories.append(trajectory)

        return True


# Alternative: Simple version for non-vectorized environments
class SimpleBackwardGraphCallback(BaseCallback):
    """Simpler callback for non-vectorized environments."""

    def __init__(self, backward_builder, env, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.backward_builder = backward_builder
        self.env = env
        self.save_freq = save_freq
        self.last_state = None
        self.last_action = None

    def _on_step(self) -> bool:
        # This is a simplified version - you'll need to adapt based on your env
        # Record the step for backward graph building
        try:
            # Get current state from the environment
            current_state = self.env.state if hasattr(self.env, 'state') else 0
            # You'll need to track action and reward from the environment
            # This will depend on your specific environment implementation

            # For demonstration - you'll need to implement proper state tracking
            if self.last_state is not None and self.last_action is not None:
                reward = 0  # You'll need to get this from the environment
                done = False  # You'll need to get this from the environment
                self.backward_builder.record_step(self.last_state, self.last_action,
                                                  reward, current_state, done)

            self.last_state = current_state
            # You'll need to track the action that was taken

        except Exception as e:
            print(f"Error in backward graph callback: {e}")

        return True


class VectorEnvBackwardGraphCallback(BaseCallback):
    """Callback for building backward graph with vectorized environments."""

    def __init__(self, backward_builder, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.backward_builder = backward_builder
        self.save_freq = save_freq
        self.step_count = 0

        # Buffers to store environment information
        self.last_states = [None] * backward_builder.n_envs
        self.last_actions = [None] * backward_builder.n_envs
        self.last_rewards = [None] * backward_builder.n_envs

    def _on_training_start(self):
        # Initialize buffers
        self.last_states = [None] * self.backward_builder.n_envs
        self.last_actions = [None] * self.backward_builder.n_envs
        self.last_rewards = [None] * self.backward_builder.n_envs

    def _on_step(self) -> bool:
        # This method is called after each step in all environments
        env_infos = []

        # For vectorized environments, we need to extract info from each environment
        for env_idx in range(self.backward_builder.n_envs):
            try:
                # Get information from the environment
                # The exact method depends on your environment implementation

                # For Stable-Baselines3 VecEnv, we can use get_attr
                if hasattr(self.model.env, 'get_attr'):
                    # Get current states
                    states = self.model.env.get_attr('state')
                    rewards = self.model.env.get_attr('last_reward', [0] * self.backward_builder.n_envs)
                    dones = self.model.env.get_attr('done')

                    # For actions, we need to track them ourselves or get from env
                    # This is tricky - we'll need to modify the environment to expose actions
                    current_action = self.last_actions[env_idx] or 0

                    env_info = {
                        'env_idx': env_idx,
                        'state': self.last_states[env_idx],  # Previous state
                        'action': current_action,
                        'reward': rewards[env_idx] if rewards else 0,
                        'next_state': states[env_idx] if states else 0,
                        'done': dones[env_idx] if dones else False
                    }

                    env_infos.append(env_info)

                    # Update buffers
                    self.last_states[env_idx] = states[env_idx] if states else 0

            except Exception as e:
                print(f"Error getting env info for env {env_idx}: {e}")
                continue

        # Record the step
        if env_infos:
            self.backward_builder.record_step(env_infos)

        # Track policy at intervals
        if self.n_calls % self.save_freq == 0:
            # You might want to save policy snapshots here
            pass

        return True


# Modified callback that works with tracking wrappers
class TrackingWrapperCallback(BaseCallback):
    """Callback that works with TrackingWrapper environments."""

    def __init__(self, backward_builder, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.backward_builder = backward_builder
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        env_infos = []

        # Collect transition info from all environments
        for env_idx in range(self.backward_builder.n_envs):
            try:
                # Get the tracking wrapper
                env = self.model.env.envs[env_idx]
                if hasattr(env, 'get_transition_info'):
                    transition_info = env.get_transition_info()
                    if transition_info:
                        env_infos.append(transition_info)
            except Exception as e:
                print(f"Error getting transition info for env {env_idx}: {e}")
                continue

        # Record the steps
        if env_infos:
            self.backward_builder.record_step(env_infos)

        return True
