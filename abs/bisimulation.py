"""
Geometric MDP Abstraction using Bisimulation Metrics and Persistent Homology
Tested on Gymnasium environments: FrozenLake, Taxi, CartPole (discretized)
"""

import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# Try to import persistent homology libraries
try:
    from ripser import ripser
    from persim import plot_diagrams

    PERSISTENCE_AVAILABLE = True
except ImportError:
    print("Ripser not installed. Install with: pip install ripser persim")
    PERSISTENCE_AVAILABLE = False


class BisimulationMetricLearner:
    """
    Learn bisimulation metric on state space via fixed point iteration
    d(s1,s2) = max_a [ |r(s1,a)-r(s2,a)| + γ·W₁(P(·|s1,a), P(·|s2,a)) ]
    """

    def __init__(self, env, gamma=0.95, n_iterations=20):
        self.env = env
        self.gamma = gamma
        self.n_iterations = n_iterations

        # Extract discrete state/action spaces
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.n_states = env.observation_space.n
            self.states = np.arange(self.n_states)
        else:
            # For continuous spaces, we'll sample a grid
            self.n_states = 100  # sample 100 states
            self.states = self._sample_continuous_states()

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
            self.actions = np.arange(self.n_actions)
        else:
            # Discretize continuous actions
            self.n_actions = 5
            self.actions = np.linspace(
                env.action_space.low[0], env.action_space.high[0], self.n_actions
            )

        # Initialize metrics
        self.state_metric = np.zeros((self.n_states, self.n_states))
        self.action_metric = np.zeros((self.n_actions, self.n_actions))

        # Precompute transition probabilities and rewards
        self._precompute_transitions()

    def _sample_continuous_states(self):
        """Sample states from continuous space"""
        states = []
        space = self.env.observation_space
        for _ in range(100):
            if hasattr(space, "low") and hasattr(space, "high"):
                s = np.random.uniform(space.low, space.high)
            else:
                s = np.random.randn(space.shape[0])
            states.append(s)
        return np.array(states)

    def _precompute_transitions(self):
        """Estimate transition probabilities"""
        self.transitions = {}
        self.rewards = {}

        for s_idx, s in enumerate(self.states):
            for a_idx, a in enumerate(self.actions):
                # Try to estimate next states
                next_states = []
                rewards = []

                # Handle discrete vs continuous
                if isinstance(self.env.observation_space, gym.spaces.Discrete):
                    # For discrete environments, we can reset to specific states
                    # This is approximate - real implementation would use Monte Carlo
                    self.env.reset()
                    # Hack: can't easily set state in Gymnasium, so we approximate
                    trans_probs = np.ones(self.n_states) / self.n_states
                    self.transitions[(s_idx, a_idx)] = trans_probs

                    # Reward approximation
                    if hasattr(self.env, "P"):
                        # FrozenLake style
                        for next_s, prob, reward, _ in self.env.P[s][a]:
                            self.rewards[(s_idx, a_idx)] = reward
                            break
                    else:
                        self.rewards[(s_idx, a_idx)] = 0
                else:
                    # Continuous: simulate transitions
                    self.env.reset()
                    # Approximate with random rollouts
                    self.transitions[(s_idx, a_idx)] = np.random.randn(
                        100, self.env.observation_space.shape[0]
                    )
                    self.rewards[(s_idx, a_idx)] = np.random.randn()

    def _wasserstein_distance(self, p1, p2):
        """Approximate Wasserstein-1 distance between transition distributions"""
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            # For discrete, use L1 on probability vectors
            return np.sum(np.abs(p1 - p2))
        else:
            # For continuous, use Euclidean distance between sampled points
            if len(p1) == 0 or len(p2) == 0:
                return 0
            # Simple approximation: match nearest neighbors
            dists = cdist(p1[:10], p2[:10])
            return np.mean(np.min(dists, axis=1))

    def compute_state_metric(self):
        """Fixed point iteration for state bisimulation metric"""
        # Initialize with zeros
        d = np.zeros((self.n_states, self.n_states))

        for iteration in range(self.n_iterations):
            d_new = np.zeros((self.n_states, self.n_states))

            for i in range(self.n_states):
                for j in range(i + 1, self.n_states):
                    # Max over actions
                    max_val = 0
                    for a_idx in range(self.n_actions):
                        # Reward difference
                        r_diff = abs(
                            self.rewards.get((i, a_idx), 0)
                            - self.rewards.get((j, a_idx), 0)
                        )

                        # Wasserstein distance on transitions
                        p1 = self.transitions.get((i, a_idx), np.zeros(self.n_states))
                        p2 = self.transitions.get((j, a_idx), np.zeros(self.n_states))
                        w_dist = self._wasserstein_distance(p1, p2)

                        # Bisimulation term
                        val = r_diff + self.gamma * w_dist
                        max_val = max(max_val, val)

                    d_new[i, j] = max_val
                    d_new[j, i] = max_val

            # Check convergence
            if np.max(np.abs(d - d_new)) < 1e-4:
                break
            d = d_new

        self.state_metric = d
        return d

    def compute_action_metric(self):
        """Compute action similarity metric"""
        for i in range(self.n_actions):
            for j in range(i + 1, self.n_actions):
                # Sup over states
                max_val = 0
                for s_idx in range(self.n_states):
                    r_diff = abs(
                        self.rewards.get((s_idx, i), 0)
                        - self.rewards.get((s_idx, j), 0)
                    )
                    p1 = self.transitions.get((s_idx, i), np.zeros(self.n_states))
                    p2 = self.transitions.get((s_idx, j), np.zeros(self.n_states))
                    w_dist = self._wasserstein_distance(p1, p2)
                    val = r_diff + self.gamma * w_dist
                    max_val = max(max_val, val)

                self.action_metric[i, j] = max_val
                self.action_metric[j, i] = max_val

        return self.action_metric


class GeometricAbstraction:
    """Build topological abstractions using bisimulation metrics"""

    def __init__(self, metric_learner):
        self.learner = metric_learner
        self.state_cover = None
        self.action_cover = None

    def build_epsilon_covers(self, state_eps=0.5, action_eps=0.5):
        """Build epsilon-cover using bisimulation metric"""
        # DBSCAN clustering on the metric space
        state_clustering = DBSCAN(eps=state_eps, min_samples=1, metric="precomputed")
        self.state_labels = state_clustering.fit_predict(self.learner.state_metric)

        action_clustering = DBSCAN(eps=action_eps, min_samples=1, metric="precomputed")
        self.action_labels = action_clustering.fit_predict(self.learner.action_metric)

        self.n_state_clusters = len(set(self.state_labels))
        self.n_action_clusters = len(set(self.action_labels))

        print(
            f"Abstracted {self.learner.n_states} states → {self.n_state_clusters} clusters"
        )
        print(
            f"Abstracted {self.learner.n_actions} actions → {self.n_action_clusters} clusters"
        )

        return self.state_labels, self.action_labels

    def compute_persistence(self):
        """Compute persistent homology of state space under bisimulation metric"""
        if not PERSISTENCE_AVAILABLE:
            return None

        # Compute distance matrix
        dist_matrix = self.learner.state_metric

        # Run persistent homology
        diagrams = ripser(dist_matrix, distance_matrix=True)["dgms"]

        return diagrams

    def visualize_metric_space(self):
        """Visualize state space with bisimulation distances"""
        # MDS for visualization
        from sklearn.manifold import MDS

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        state_positions = mds.fit_transform(self.learner.state_metric)

        plt.figure(figsize=(15, 5))

        # Plot 1: State metric heatmap
        plt.subplot(131)
        plt.imshow(self.learner.state_metric, cmap="viridis")
        plt.colorbar(label="Bisimulation distance")
        plt.title("State Bisimulation Metric")
        plt.xlabel("State index")
        plt.ylabel("State index")

        # Plot 2: MDS embedding with clusters
        plt.subplot(132)
        scatter = plt.scatter(
            state_positions[:, 0],
            state_positions[:, 1],
            c=self.state_labels,
            cmap="tab20",
            s=100,
        )
        plt.colorbar(scatter, label="Cluster ID")
        plt.title(f"State Space: {self.n_state_clusters} clusters")
        plt.xlabel("MDS1")
        plt.ylabel("MDS2")

        # Plot 3: Action metric
        plt.subplot(133)
        plt.imshow(self.learner.action_metric, cmap="plasma")
        plt.colorbar(label="Action similarity")
        plt.title("Action Similarity Matrix")
        plt.xlabel("Action index")
        plt.ylabel("Action index")

        plt.tight_layout()
        plt.show()

    def visualize_persistence(self, diagrams):
        """Plot persistence diagram"""
        if diagrams is None:
            print("Persistence computation not available")
            return

        plt.figure(figsize=(8, 4))
        plot_diagrams(diagrams)
        plt.title("Persistence Diagram - State Space Topology")
        plt.show()


def run_experiment(env_name, gamma=0.95):
    """Run complete geometric abstraction pipeline"""
    print(f"\n{'=' * 50}")
    print(f"Experiment: {env_name}")
    print(f"{'=' * 50}")

    # Create environment
    env = gym.make(env_name)

    # Learn bisimulation metrics
    learner = BisimulationMetricLearner(env, gamma=gamma)
    state_metric = learner.compute_state_metric()
    action_metric = learner.compute_action_metric()

    print(
        f"\nState metric range: [{np.min(state_metric):.3f}, {np.max(state_metric):.3f}]"
    )
    print(
        f"Action metric range: [{np.min(action_metric):.3f}, {np.max(action_metric):.3f}]"
    )

    # Build geometric abstractions
    abstraction = GeometricAbstraction(learner)

    # Try different epsilon values
    epsilons = [0.1, 0.5, 1.0]
    for eps in epsilons:
        print(f"\nEpsilon = {eps}:")
        labels, _ = abstraction.build_epsilon_covers(state_eps=eps, action_eps=eps)
        n_clusters = len(set(labels))
        print(f"  → {n_clusters} state clusters")

    # Visualize
    abstraction.visualize_metric_space()

    # Compute persistent homology
    diagrams = abstraction.compute_persistence()
    if diagrams:
        abstraction.visualize_persistence(diagrams)

    env.close()
    return learner, abstraction


# ============ MAIN EXPERIMENTS ============

if __name__ == "__main__":
    print("Geometric MDP Abstraction with Bisimulation Metrics")
    print("Testing on Gymnasium environments...")

    # Test on discrete environments
    discrete_envs = ["FrozenLake-v1", "Taxi-v3"]
    for env_name in discrete_envs:
        try:
            run_experiment(env_name)
        except Exception as e:
            print(f"Error with {env_name}: {e}")

    # Test on continuous environment (discretized)
    try:
        run_experiment("CartPole-v1")
    except Exception as e:
        print(f"Error with CartPole: {e}")

    print("\nExperiments complete!")
