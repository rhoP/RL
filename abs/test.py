import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import gudhi as gd
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class ReplayBuffer:
    """Experience replay buffer for storing trajectories"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class LowLevelPolicy(nn.Module):
    """Lower level policy π_L that generates trajectories"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Priority bias network (influenced by high-level policy)
        self.priority_bias = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for abstract state value
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self, state: torch.Tensor, abstract_value: Optional[torch.Tensor] = None
    ):
        base_actions = self.network(state)

        if abstract_value is not None:
            # Apply priority bias from high-level policy
            combined = torch.cat([state, abstract_value.unsqueeze(-1)], dim=-1)
            bias = self.priority_bias(combined)
            return (
                base_actions + bias
            )  # Prioritize actions toward valuable abstract states

        return base_actions


class HighLevelPolicy(nn.Module):
    """Higher level policy π_U that learns value function on abstract states"""

    def __init__(self, abstract_state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Linear(abstract_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output value V(s_abstract)
        )

    def forward(self, abstract_state: torch.Tensor):
        return self.value_network(abstract_state)


class TopologicalStateAggregator:
    """Forms simplicial complex of abstract states using persistent homology"""

    def __init__(self, epsilon: float = 0.1, min_cluster_size: int = 5):
        self.epsilon = epsilon  # Connectivity threshold
        self.min_cluster_size = min_cluster_size
        self.abstract_states = {}  # Mapping from original state indices to abstract state IDs
        self.state_values = {}  # Values for abstract states
        self.stabilized = False

    def compute_distance_matrix(
        self, states: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted distance matrix considering both state similarity and value difference
        """
        # Spatial distances
        spatial_dist = squareform(pdist(states, metric="euclidean"))

        # Value differences (normalized)
        value_diff = np.abs(values[:, None] - values[None, :])
        value_diff = value_diff / (value_diff.max() + 1e-8)

        # Combined metric: states close in space AND similar in value
        # The weighting can be adjusted based on problem domain
        alpha = 0.7  # Weight for spatial proximity
        beta = 0.3  # Weight for value similarity

        combined_dist = alpha * spatial_dist + beta * value_diff
        return combined_dist

    def build_simplicial_complex(
        self, states: np.ndarray, values: np.ndarray
    ) -> List[List[int]]:
        """
        Build Vietoris-Rips complex using persistent homology
        """
        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(states, values)

        # Create Vietoris-Rips complex
        rips_complex = gd.RipsComplex(distance_matrix=dist_matrix)

        # Build simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

        # Compute persistence
        persistence = simplex_tree.persistence()

        # Extract clusters based on persistence
        clusters = []

        # Use persistence to identify stable connected components
        # Get the 0-dimensional persistence diagram (connected components)
        diag = simplex_tree.persistence_intervals_in_dimension(0)

        # Find persistent clusters (those that survive beyond epsilon)
        for i, (birth, death) in enumerate(diag):
            if death > birth + self.epsilon:  # Persistent component
                # Get vertices in this component (simplified - in practice need to track vertices)
                cluster = [i]  # Simplified: each persistent component gets its vertices
                if len(cluster) >= self.min_cluster_size:
                    clusters.append(cluster)

        # Alternative: Use hierarchical clustering with persistence-based threshold
        if len(clusters) == 0:
            # Fallback to hierarchical clustering
            linkage_matrix = linkage(dist_matrix, method="ward")
            cluster_labels = fcluster(
                linkage_matrix, t=self.epsilon, criterion="distance"
            )

            for cluster_id in np.unique(cluster_labels):
                cluster = np.where(cluster_labels == cluster_id)[0].tolist()
                if len(cluster) >= self.min_cluster_size:
                    clusters.append(cluster)

        return clusters

    def update_abstract_states(self, trajectories: List[Tuple]):
        """
        Update abstract state mapping based on new trajectories
        """
        if len(trajectories) < self.min_cluster_size:
            return self.stabilized

        # Extract states and their values from trajectories
        states = []
        values = []

        for trajectory in trajectories:
            if len(trajectory) >= 4:  # state, action, reward, next_state, done
                states.append(trajectory[0])
                # Use cumulative future reward as state value
                reward = trajectory[2]
                values.append(reward)

        if len(states) < self.min_cluster_size:
            return self.stabilized

        states = np.array(states)
        values = np.array(values)

        # Build clusters
        clusters = self.build_simplicial_complex(states, values)

        # Create new abstract state mapping
        new_abstract_states = {}
        for cluster_idx, cluster in enumerate(clusters):
            abstract_state_id = f"AS_{cluster_idx}"
            for state_idx in cluster:
                new_abstract_states[state_idx] = abstract_state_id
            # Store average value for this abstract state
            self.state_values[abstract_state_id] = np.mean(values[cluster])

        # Check if mapping has stabilized (minimal changes)
        if self.abstract_states:
            changes = sum(
                1
                for k in new_abstract_states
                if k in self.abstract_states
                and new_abstract_states[k] != self.abstract_states[k]
            )
            change_ratio = changes / len(new_abstract_states)
            self.stabilized = change_ratio < 0.1  # Less than 10% change

        self.abstract_states = new_abstract_states
        return self.stabilized

    def get_abstract_state(self, state_idx: int) -> Optional[str]:
        """Get abstract state ID for a given state index"""
        return self.abstract_states.get(state_idx)

    def get_abstract_state_value(self, abstract_state_id: str) -> float:
        """Get value of an abstract state"""
        return self.state_values.get(abstract_state_id, 0.0)


class HierarchicalAgent:
    """Main hierarchical agent combining both policies"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        abstract_state_dim: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update parameter
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Initialize policies
        self.pi_l = LowLevelPolicy(state_dim, action_dim)
        self.pi_u = HighLevelPolicy(abstract_state_dim)
        self.pi_u_target = HighLevelPolicy(abstract_state_dim)
        self.pi_u_target.load_state_dict(self.pi_u.state_dict())

        # Optimizers
        self.optimizer_l = optim.Adam(self.pi_l.parameters(), lr=lr)
        self.optimizer_u = optim.Adam(self.pi_u.parameters(), lr=lr)

        # Replay buffers
        self.buffer_l = ReplayBuffer(capacity=100000)
        self.buffer_u = ReplayBuffer(capacity=50000)

        # Topological aggregator
        self.aggregator = TopologicalStateAggregator(epsilon=0.1)

        # Trajectory storage for persistent homology
        self.trajectory_buffer = []
        self.max_trajectories = 1000

        # Abstract state embedding (simplified - could use learned embedding)
        self.abstract_state_embedding = nn.Linear(state_dim, abstract_state_dim)

    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Get action from lower policy with high-level guidance
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get abstract state value guidance if available
        abstract_value = None
        if hasattr(self, "current_abstract_value"):
            abstract_value = torch.FloatTensor([self.current_abstract_value])

        with torch.no_grad():
            action_logits = self.pi_l(state_tensor, abstract_value)

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return torch.argmax(action_logits).item()

    def get_abstract_value(self, state: np.ndarray) -> float:
        """
        Get high-level policy's value for the abstract state containing the current state
        """
        # Find which abstract state this state belongs to (simplified - need proper mapping)
        # In practice, you'd maintain a mapping from state vectors to abstract state IDs
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        abstract_embedding = self.abstract_state_embedding(state_tensor)

        with torch.no_grad():
            value = self.pi_u(abstract_embedding)

        return value.item()

    def update_high_level(self, batch_size: int = 64):
        """
        Update high-level policy using abstract state values
        """
        if len(self.buffer_u) < batch_size:
            return

        states, _, rewards, next_states, dones = self.buffer_u.sample(batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get abstract state embeddings
        abstract_states = self.abstract_state_embedding(states)
        abstract_next_states = self.abstract_state_embedding(next_states)

        # Compute target values
        with torch.no_grad():
            next_values = self.pi_u_target(abstract_next_states)
            target_values = rewards + (1 - dones) * self.gamma * next_values

        # Compute current values
        current_values = self.pi_u(abstract_states)

        # Update high-level policy
        loss = nn.MSELoss()(current_values, target_values)
        self.optimizer_u.zero_grad()
        loss.backward()
        self.optimizer_u.step()

        # Soft update target network
        for target_param, param in zip(
            self.pi_u_target.parameters(), self.pi_u.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update_low_level(self, batch_size: int = 64):
        """
        Update low-level policy with high-level guidance
        """
        if len(self.buffer_l) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer_l.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get high-level guidance values for current and next states
        current_abstract_values = []
        next_abstract_values = []

        for i in range(len(states)):
            current_abstract_values.append(self.get_abstract_value(states[i].numpy()))
            if not dones[i]:
                next_abstract_values.append(
                    self.get_abstract_value(next_states[i].numpy())
                )
            else:
                next_abstract_values.append(0.0)

        current_abstract_values = torch.FloatTensor(current_abstract_values)
        next_abstract_values = torch.FloatTensor(next_abstract_values)

        # Compute low-level policy loss with high-level guidance
        action_logits = self.pi_l(states, current_abstract_values)
        action_probs = torch.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # High-level guidance bonus: encourage actions leading to high-value abstract states
        guidance_bonus = next_abstract_values * 0.1  # Weighted guidance

        # Compute advantages with guidance
        with torch.no_grad():
            # Target values include both immediate reward and future abstract state value
            targets = rewards + self.gamma * next_abstract_values * (1 - dones)

        advantages = targets - action_probs.mean(dim=-1).detach()

        # Policy gradient loss with guidance
        policy_loss = -(log_probs * advantages).mean()

        self.optimizer_l.zero_grad()
        policy_loss.backward()
        self.optimizer_l.step()

    def add_trajectory(self, state, action, reward, next_state, done):
        """Add a trajectory step to buffers"""
        self.buffer_l.push(state, action, reward, next_state, done)
        self.buffer_u.push(state, action, reward, next_state, done)

        # Store for topological analysis
        self.trajectory_buffer.append((state, action, reward, next_state, done))
        if len(self.trajectory_buffer) > self.max_trajectories:
            self.trajectory_buffer.pop(0)

    def update_topology(self):
        """Update topological state aggregation"""
        if len(self.trajectory_buffer) >= 100:  # Minimum trajectories for analysis
            stabilized = self.aggregator.update_abstract_states(self.trajectory_buffer)
            return stabilized
        return False


def train_hierarchical_agent(env_name: str = "CartPole-v1", episodes: int = 1000):
    """
    Training loop for the hierarchical agent
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = HierarchicalAgent(state_dim, action_dim)

    episode_rewards = []
    topology_stabilized = False

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action from agent
            action = agent.get_action(state, epsilon=max(0.1, 0.5 - episode / 500))

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store trajectory
            agent.add_trajectory(state, action, reward, next_state, done)

            # Update policies
            agent.update_low_level()
            agent.update_high_level()

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

        # Update topological aggregation periodically
        if episode % 50 == 0 and not topology_stabilized:
            stabilized = agent.update_topology()
            if stabilized:
                print(f"Episode {episode}: Topological aggregation stabilized")
                topology_stabilized = stabilized

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
            if avg_reward >= env.spec.reward_threshold:
                print(f"Solved at episode {episode + 1}!")
                break

    env.close()
    return agent, episode_rewards


# Example usage and testing
if __name__ == "__main__":
    # Train the hierarchical agent
    print("Training Hierarchical Agent with Topological State Aggregation...")
    agent, rewards = train_hierarchical_agent("CartPole-v1", episodes=500)

    # Plot results (optional)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()

    print("Training completed!")
