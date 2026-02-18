import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict, Set
import networkx as nx
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import eigsh
import warnings

warnings.filterwarnings("ignore")

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class StateGraphBuilder:
    """
    Builds and maintains a state graph from the replay buffer with weighted edges
    based on transition probabilities.
    """

    def __init__(self, state_dim: int, similarity_threshold: float = 0.85):
        self.state_dim = state_dim
        self.similarity_threshold = similarity_threshold
        self.graph = nx.DiGraph()
        self.state_cache = {}  # Store raw states for similarity computation
        self.transition_counts = {}  # Count transitions for probability calculation

    def _compute_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute cosine similarity between two states."""
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(state1, state2) / (norm1 * norm2)

    def _get_or_create_node(self, state: np.ndarray) -> int:
        """Get existing node ID or create new node for state."""
        state_tuple = tuple(state.flatten())

        # Check if similar state exists
        for node_id, cached_state in self.state_cache.items():
            if (
                self._compute_similarity(state, cached_state)
                > self.similarity_threshold
            ):
                return node_id

        # Create new node
        node_id = len(self.graph.nodes)
        self.graph.add_node(node_id)
        self.state_cache[node_id] = state
        return node_id

    def add_transition(self, state: np.ndarray, next_state: np.ndarray):
        """Add a transition to the graph."""
        state_node = self._get_or_create_node(state)
        next_node = self._get_or_create_node(next_state)

        # Update transition counts
        key = (state_node, next_node)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

    def build_weighted_graph(self) -> nx.DiGraph:
        """Build weighted graph based on transition probabilities."""
        weighted_graph = nx.DiGraph()

        # Copy nodes
        weighted_graph.add_nodes_from(self.graph.nodes)

        # Add weighted edges based on transition probabilities
        for (src, dst), count in self.transition_counts.items():
            # Calculate transition probability from src to dst
            total_out = sum(
                c for (s, _), c in self.transition_counts.items() if s == src
            )
            if total_out > 0:
                prob = count / total_out
                weighted_graph.add_edge(src, dst, weight=prob, count=count)

        return weighted_graph


class SpectralSmoother:
    """
    Applies spectral smoothing to the state graph using graph Laplacian.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.laplacian_eigenvals = None
        self.laplacian_eigenvecs = None

    def compute_laplacian(
        self, graph: nx.Graph
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute graph Laplacian and its eigendecomposition."""
        # Get adjacency matrix with weights
        n_nodes = len(graph.nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            adj_matrix[u, v] = weight
            if not graph.is_directed():
                adj_matrix[v, u] = weight

        # Compute degree matrix
        degree_matrix = np.diag(adj_matrix.sum(axis=1))

        # Compute Laplacian (normalized)
        laplacian = degree_matrix - adj_matrix

        # Compute eigendecomposition (smallest eigenvalues)
        try:
            eigenvals, eigenvecs = eigsh(
                laplacian, k=min(self.n_components, n_nodes - 1), which="SM"
            )
        except:
            # Fallback to regular eigenvalue computation
            eigenvals, eigenvecs = np.linalg.eigh(laplacian)
            eigenvals = eigenvals[: self.n_components]
            eigenvecs = eigenvecs[:, : self.n_components]

        self.laplacian_eigenvals = eigenvals
        self.laplacian_eigenvecs = eigenvecs

        return laplacian, eigenvals, eigenvecs

    def smooth_state_values(self, state_values: np.ndarray) -> np.ndarray:
        """Apply spectral smoothing to state values."""
        if self.laplacian_eigenvecs is None:
            raise ValueError("Must compute Laplacian first")

        # Project onto eigenbasis
        coefficients = self.laplacian_eigenvecs.T @ state_values

        # Smooth by truncating high-frequency components
        # (already truncated by using only smallest eigenvalues)
        smoothed_values = self.laplacian_eigenvecs @ coefficients

        return smoothed_values


class PersistentConnectivityFilter:
    """
    Applies persistent connectivity filtering by thresholding and keeping
    the connected component containing the goal state.
    """

    def __init__(self, thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]):
        self.thresholds = thresholds
        self.persistent_components = {}

    def filter_by_threshold(self, graph: nx.Graph, threshold: float) -> nx.Graph:
        """Filter graph edges by weight threshold."""
        filtered_graph = nx.Graph()

        # Copy all nodes
        filtered_graph.add_nodes_from(graph.nodes)

        # Add edges with weight above threshold
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 0)
            if weight >= threshold:
                filtered_graph.add_edge(u, v, weight=weight)

        return filtered_graph

    def find_goal_component(self, graph: nx.Graph, goal_node: int) -> Set[int]:
        """Find the connected component containing the goal node."""
        if goal_node not in graph:
            return {goal_node}

        # Use BFS to find connected component
        visited = set()
        queue = deque([goal_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            neighbors = list(graph.neighbors(node))
            queue.extend(neighbors)

        return visited

    def compute_persistence(
        self, graphs: List[nx.Graph], goal_node: int
    ) -> Dict[int, float]:
        """Compute persistence scores for nodes across thresholds."""
        node_persistence = {}

        for threshold in self.thresholds:
            filtered_graph = self.filter_by_threshold(graphs[0], threshold)
            component = self.find_goal_component(filtered_graph, goal_node)

            # Update persistence scores
            for node in component:
                node_persistence[node] = node_persistence.get(node, 0) + 1

        # Normalize by number of thresholds
        for node in node_persistence:
            node_persistence[node] /= len(self.thresholds)

        return node_persistence


class BackwardTransferPropagation:
    """
    Implements backward transfer propagation from goal states.
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.state_values = {}

    def propagate_backward(
        self, graph: nx.DiGraph, goal_node: int, goal_value: float = 1.0
    ) -> Dict[int, float]:
        """Propagate values backward from goal through the graph."""
        # Initialize values
        values = {node: 0.0 for node in graph.nodes}
        values[goal_node] = goal_value

        # Bellman-like backward propagation
        changed = True
        max_iterations = 1000
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            new_values = values.copy()

            # For each node, update based on forward transitions
            for node in graph.nodes:
                if node == goal_node:
                    continue

                # Look at all successors
                successors = list(graph.successors(node))
                if not successors:
                    continue

                # Calculate expected value from successors
                expected_value = 0
                total_weight = 0

                for succ in successors:
                    weight = graph[node][succ].get("weight", 1.0)
                    expected_value += weight * values[succ]
                    total_weight += weight

                if total_weight > 0:
                    expected_value /= total_weight
                    new_value = self.gamma * expected_value

                    if abs(new_value - values[node]) > 1e-6:
                        new_values[node] = new_value
                        changed = True

            values = new_values
            iteration += 1

        self.state_values = values
        return values


class CombinedRLAlgorithm:
    """
    Main class combining all three techniques for reinforcement learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        buffer_size: int = 10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Initialize components
        self.graph_builder = StateGraphBuilder(state_dim)
        self.spectral_smoother = SpectralSmoother()
        self.connectivity_filter = PersistentConnectivityFilter()
        self.backward_propagator = BackwardTransferPropagation(gamma)

        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)

        # Neural network for Q-learning
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Training parameters
        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0

    def _build_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer and update graph."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

        # Update state graph
        self.graph_builder.add_transition(state, next_state)

    def compute_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values for state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).numpy().flatten()
        return q_values

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.compute_q_values(state)
        return np.argmax(q_values)

    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor(np.array([e.action for e in batch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor(np.array([e.done for e in batch]))

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def apply_combined_techniques(self, goal_state: np.ndarray) -> Dict:
        """
        Apply the combined backward transfer propagation, spectral smoothing,
        and persistent connectivity filtering.
        """
        # Build weighted graph from replay buffer
        weighted_graph = self.graph_builder.build_weighted_graph()

        if len(weighted_graph.nodes) == 0:
            return {}

        # Find goal node
        goal_node = self.graph_builder._get_or_create_node(goal_state)

        # Step 1: Backward transfer propagation
        print("Applying backward transfer propagation...")
        backward_values = self.backward_propagator.propagate_backward(
            weighted_graph, goal_node
        )

        # Step 2: Spectral smoothing
        print("Applying spectral smoothing...")
        # Convert to undirected for spectral analysis
        undirected_graph = weighted_graph.to_undirected()
        laplacian, eigenvals, eigenvecs = self.spectral_smoother.compute_laplacian(
            undirected_graph
        )

        # Smooth the backward values
        value_array = np.array(
            [backward_values.get(i, 0) for i in range(len(undirected_graph.nodes))]
        )
        smoothed_values = self.spectral_smoother.smooth_state_values(value_array)

        # Step 3: Persistent connectivity filtering
        print("Applying persistent connectivity filtering...")
        persistence_scores = self.connectivity_filter.compute_persistence(
            [undirected_graph], goal_node
        )

        # Apply different thresholds to get persistent component
        persistent_component = set()
        for threshold in self.connectivity_filter.thresholds:
            filtered_graph = self.connectivity_filter.filter_by_threshold(
                undirected_graph, threshold
            )
            component = self.connectivity_filter.find_goal_component(
                filtered_graph, goal_node
            )
            persistent_component.update(component)

        # Combine results
        results = {
            "backward_values": backward_values,
            "smoothed_values": {
                i: smoothed_values[i] for i in range(len(smoothed_values))
            },
            "persistence_scores": persistence_scores,
            "persistent_component": persistent_component,
            "graph": weighted_graph,
            "laplacian_eigenvals": eigenvals,
            "laplacian_eigenvecs": eigenvecs,
        }

        return results


def create_sample_environment(size=5):
    """Create a simple grid world environment for demonstration."""

    class GridWorld:
        def __init__(self, size=5):
            self.size = size
            self.state = np.array([0, 0])
            self.goal = np.array([size - 1, size - 1])

        def reset(self):
            self.state = np.array([0, 0])
            return self.state.copy()

        def step(self, action):
            # Actions: 0=up, 1=right, 2=down, 3=left
            next_state = self.state.copy()

            if action == 0 and self.state[0] > 0:
                next_state[0] -= 1
            elif action == 1 and self.state[1] < self.size - 1:
                next_state[1] += 1
            elif action == 2 and self.state[0] < self.size - 1:
                next_state[0] += 1
            elif action == 3 and self.state[1] > 0:
                next_state[1] -= 1

            # Calculate reward
            reward = 1.0 if np.array_equal(next_state, self.goal) else -0.01
            done = np.array_equal(next_state, self.goal)

            self.state = next_state
            return next_state, reward, done

    return GridWorld()


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Combined RL Algorithm with Backward Transfer Propagation,")
    print("Spectral Smoothing, and Persistent Connectivity Filtering")
    print("=" * 60)

    # Create environment
    env = create_sample_environment(size=5)
    state_dim = 2  # x, y coordinates
    action_dim = 4  # up, right, down, left

    # Initialize algorithm
    agent = CombinedRLAlgorithm(state_dim, action_dim)

    # Training loop
    print("\nTraining the agent...")
    n_episodes = 100

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, epsilon=max(0.1, 0.5 - episode / 200))

            # Take step
            next_state, reward, done = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()

            total_reward += reward
            state = next_state

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

    # Apply combined techniques
    print("\n" + "=" * 60)
    print("Applying Combined Techniques")
    print("=" * 60)

    goal_state = np.array([4, 4])  # Bottom-right corner
    results = agent.apply_combined_techniques(goal_state)

    # Display results
    if results:
        print(f"\nGraph Statistics:")
        print(f"  Number of nodes: {len(results['graph'].nodes)}")
        print(f"  Number of edges: {len(results['graph'].edges)}")

        print(f"\nBackward Propagation Results:")
        top_nodes = sorted(
            results["backward_values"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        for node, value in top_nodes:
            print(f"  Node {node}: value = {value:.4f}")

        print(f"\nSpectral Smoothing Results:")
        print(f"  Top 5 eigenvalues: {results['laplacian_eigenvals'][:5]}")

        print(f"\nPersistent Connectivity Filtering:")
        print(f"  Size of persistent component: {len(results['persistent_component'])}")

        print(f"\nTop Persistence Scores:")
        top_persistent = sorted(
            results["persistence_scores"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        for node, score in top_persistent:
            print(f"  Node {node}: persistence = {score:.4f}")

        # Demonstrate how to use the results for improved exploration
        print("\n" + "=" * 60)
        print("Using Results for Improved Exploration")
        print("=" * 60)

        # Create exploration bonus based on persistence
        def get_exploration_bonus(state):
            node = agent.graph_builder._get_or_create_node(state)
            persistence = results["persistence_scores"].get(node, 0)
            return 0.1 * persistence  # Bonus for persistent states

        # Test exploration bonus
        test_state = np.array([2, 2])
        bonus = get_exploration_bonus(test_state)
        print(f"Exploration bonus for state {test_state}: {bonus:.4f}")

    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
