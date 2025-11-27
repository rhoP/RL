import numpy as np
import pickle
from collections import defaultdict, deque
import gymnasium as gym
from typing import List, Dict, Any, Optional


class TrajectoryContainer:
    def __init__(self, max_trajectories=1000, max_steps_per_trajectory=1000):
        """
        Storage container for complete trajectories.
        
        Args:
            max_trajectories: Maximum number of trajectories to store
            max_steps_per_trajectory: Maximum steps per trajectory (for pre-allocation)
        """
        self.max_trajectories = max_trajectories
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.trajectories = deque(maxlen=max_trajectories)
        self.current_trajectory = None
        self.state_dim = None
        self.action_dim = None
        self.info_keys = None
        
    def new_trajectory(self, initial_state: np.ndarray, info: Optional[Dict] = None):
        """Start recording a new trajectory"""
        if self.current_trajectory is not None:
            # Save the current trajectory if one exists
            self.finalize_trajectory()
            
        self.current_trajectory = {
            'states': [initial_state],
            'actions': [],
            'rewards': [],
            'infos': [info] if info is not None else [],
            'terminated': False,
            'truncated': False
        }
        
        # Update dimensions if not set
        if self.state_dim is None:
            self.state_dim = initial_state.shape[0] if hasattr(initial_state, 'shape') else len(initial_state)
    
    def add_step(self, action: np.ndarray, next_state: np.ndarray, reward: float, 
                 info: Optional[Dict] = None, terminated: bool = False, 
                 truncated: bool = False):
        """Add a step to the current trajectory"""
        if self.current_trajectory is None:
            raise ValueError("No active trajectory. Call new_trajectory first.")
            
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['states'].append(next_state)
        self.current_trajectory['rewards'].append(reward)

        if info is not None:
            self.current_trajectory['infos'].append(info)
            
        # Update action dimension if not set
        if self.action_dim is None and len(action) > 0:
            self.action_dim = action.shape[0] if hasattr(action, 'shape') else len(action)
            
        # Check if trajectory should be finalized
        if terminated or truncated:
            self.current_trajectory['terminated'] = terminated
            self.current_trajectory['truncated'] = truncated
            self.finalize_trajectory()
    
    def finalize_trajectory(self):
        """Finalize and store the current trajectory"""
        if self.current_trajectory is None:
            return
            
        trajectory = {
            'states': np.array(self.current_trajectory['states']),
            'actions': np.array(self.current_trajectory['actions']),
            'rewards': np.array(self.current_trajectory['rewards']),
            'infos': self.current_trajectory['infos'],
            'terminated': self.current_trajectory['terminated'],
            'truncated': self.current_trajectory['truncated'],
            'length': len(self.current_trajectory['actions']),
            'total_reward': np.sum(self.current_trajectory['rewards'])
        }
        
        self.trajectories.append(trajectory)
        self.current_trajectory = None
    
    def add_complete_trajectory(self, states: List[np.ndarray], actions: List[np.ndarray], 
                               rewards: List[float], 
                               infos: Optional[List[Dict]] = None,
                               terminated: bool = False, truncated: bool = False):
        """Add a complete trajectory at once"""
        if len(states) != len(actions) + 1:
            raise ValueError("Number of states should be number of actions + 1")
            
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'infos': infos if infos is not None else [],
            'terminated': terminated,
            'truncated': truncated,
            'length': len(actions),
            'total_reward': np.sum(rewards)
        }
        
        self.trajectories.append(trajectory)
        
        # Update dimensions
        if self.state_dim is None and len(states) > 0:
            self.state_dim = states[0].shape[0] if hasattr(states[0], 'shape') else len(states[0])
        if self.action_dim is None and len(actions) > 0:
            self.action_dim = actions[0].shape[0] if hasattr(actions[0], 'shape') else len(actions[0])
    
    def get_trajectory(self, index: int) -> Dict[str, Any]:
        """Get a specific trajectory by index"""
        return self.trajectories[index]
    
    def get_recent_trajectories(self, n: int) -> List[Dict[str, Any]]:
        """Get the n most recent trajectories"""
        return list(self.trajectories)[-n:]
    
    def get_high_reward_trajectories(self, threshold: float) -> List[Dict[str, Any]]:
        """Get trajectories with total reward above threshold"""
        return [traj for traj in self.trajectories if traj['total_reward'] >= threshold]
    
    def get_state_action_pairs(self, max_samples: Optional[int] = None) -> np.ndarray:
        """Extract state-action pairs from all trajectories"""
        state_action_pairs = []
        
        for traj in self.trajectories:
            states = traj['states'][:-1]  # All states except the last one
            actions = traj['actions']
            
            for state, action in zip(states, actions):
                state_flat = state.flatten() if hasattr(state, 'flatten') else state
                action_flat = action.flatten() if hasattr(action, 'flatten') else action
                
                state_action = np.concatenate([state_flat, action_flat])
                state_action_pairs.append(state_action)
                
                if max_samples and len(state_action_pairs) >= max_samples:
                    return np.array(state_action_pairs)
        
        return np.array(state_action_pairs)
    
    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Compute statistics over all stored trajectories"""
        if not self.trajectories:
            return {}
            
        total_rewards = [traj['total_reward'] for traj in self.trajectories]
        lengths = [traj['length'] for traj in self.trajectories]
        
        return {
            'num_trajectories': len(self.trajectories),
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'completion_rate': np.mean([1 for traj in self.trajectories if traj['terminated']])
        }
    
    def save(self, filename: str):
        """Save trajectories to file"""
        data = {
            'trajectories': list(self.trajectories),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_trajectories': self.max_trajectories
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filename: str):
        """Load trajectories from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.trajectories = deque(data['trajectories'], maxlen=data.get('max_trajectories', self.max_trajectories))
            self.state_dim = data['state_dim']
            self.action_dim = data['action_dim']
    
    def clear(self):
        """Clear all trajectories"""
        self.trajectories.clear()
        self.current_trajectory = None
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, index):
        return self.trajectories[index]


# Integration with SB3 training
class TrajectoryCollectorCallback:
    def __init__(self, storage: TrajectoryContainer):
        self.storage = storage
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_infos = []
        
    def on_training_start(self):
        self.current_episode_states = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_infos = []
    
    def on_rollout_start(self):
        pass
    
    def on_step(self, locals_, globals_):
        # This would need to be adapted based on SB3's internal structure
        # SB3 doesn't provide a straightforward way to access individual steps in callbacks
        pass
    
    def collect_complete_episode(self, states, actions, rewards, infos, terminated, truncated):
        """Method to be called when a complete episode is available"""
        self.storage.add_complete_trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            infos=infos,
            terminated=terminated,
            truncated=truncated
        )

