import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class MDPDeterminismValidator:
    """
    Empirically validate how close to deterministic an MDP is when we consider
    the state space partitioned into balls of varying radius.
    """

    def __init__(self, env, policy=None, n_samples=10000):
        """
        Args:
            env: Gymnasium environment
            policy: Optional policy (if None, uses random actions)
            n_samples: Number of state transitions to sample
        """
        self.env = env
        self.policy = policy
        self.n_samples = n_samples

        # Get state space info
        self.state_dim = env.observation_space.shape[0]
        self.state_low = env.observation_space.low
        self.state_high = env.observation_space.high

        # Replace infinities with reasonable bounds
        self.state_low = np.where(np.isfinite(self.state_low), self.state_low, -10)
        self.state_high = np.where(np.isfinite(self.state_high), self.state_high, 10)

        # Store collected data
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []

    def collect_transitions(self, n_samples=None):
        """Collect random transitions from the environment."""
        if n_samples is None:
            n_samples = self.n_samples

        print(f"Collecting {n_samples} transitions...")

        for _ in tqdm(range(n_samples)):
            # Reset environment
            state, _ = self.env.reset()

            # Run one step
            if self.policy is not None:
                action, _ = self.policy.predict(state, deterministic=False)
            else:
                action = self.env.action_space.sample()

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Store transition
            self.states.append(state.copy())
            self.actions.append(action.copy() if hasattr(action, "copy") else action)
            self.next_states.append(next_state.copy())
            self.rewards.append(reward)

            if terminated or truncated:
                continue

        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.next_states = np.array(self.next_states)
        self.rewards = np.array(self.rewards)

        print(f"Collected {len(self.states)} transitions")

    def build_state_tree(self, states=None):
        """Build KD-tree for efficient ball queries."""
        if states is None:
            states = self.states
        return KDTree(states)

    def compute_ball_statistics(self, radius, min_points_per_ball=5):
        """
        For a given radius, compute statistics about transition determinism.

        Args:
            radius: Radius of state balls
            min_points_per_ball: Minimum points to consider a ball

        Returns:
            dict: Statistics about transition determinism
        """
        if len(self.states) == 0:
            raise ValueError(
                "No transitions collected. Run collect_transitions() first."
            )

        tree = self.build_state_tree()

        # Find balls around each state
        ball_stats = {
            "state_indices": [],
            "n_points": [],
            "next_state_variance": [],
            "reward_variance": [],
            "next_state_entropy": [],
            "next_state_radius": [],
            "is_deterministic": [],
        }

        # Use a subset of states as ball centers to avoid overlap
        # Sample every 10th state or so
        center_indices = np.random.choice(
            len(self.states), min(1000, len(self.states)), replace=False
        )

        for idx in tqdm(center_indices, desc=f"Analyzing balls (r={radius:.3f})"):
            center_state = self.states[idx]

            # Find all states within radius
            indices = tree.query_ball_point(center_state, radius)

            if len(indices) >= min_points_per_ball:
                # Get next states for all points in ball
                next_states_in_ball = self.next_states[indices]
                rewards_in_ball = self.rewards[indices]

                # Compute variance of next states
                next_state_var = np.var(next_states_in_ball, axis=0).mean()

                # Compute variance of rewards
                reward_var = np.var(rewards_in_ball)

                # Compute entropy of next state distribution
                # Discretize next states for entropy calculation
                next_state_entropy = self._estimate_entropy(next_states_in_ball)

                # Compute radius of next state distribution
                next_state_center = np.mean(next_states_in_ball, axis=0)
                next_state_radii = np.linalg.norm(
                    next_states_in_ball - next_state_center, axis=1
                )
                mean_next_radius = np.mean(next_state_radii)

                # Consider deterministic if next state variance is small
                is_deterministic = next_state_var < 0.01 * radius

                ball_stats["state_indices"].append(indices)
                ball_stats["n_points"].append(len(indices))
                ball_stats["next_state_variance"].append(next_state_var)
                ball_stats["reward_variance"].append(reward_var)
                ball_stats["next_state_entropy"].append(next_state_entropy)
                ball_stats["next_state_radius"].append(mean_next_radius)
                ball_stats["is_deterministic"].append(is_deterministic)

        # Aggregate statistics
        results = {
            "radius": radius,
            "n_balls": len(ball_stats["n_points"]),
            "mean_points_per_ball": np.mean(ball_stats["n_points"]),
            "mean_next_state_variance": np.mean(ball_stats["next_state_variance"]),
            "std_next_state_variance": np.std(ball_stats["next_state_variance"]),
            "mean_reward_variance": np.mean(ball_stats["reward_variance"]),
            "mean_next_state_entropy": np.mean(ball_stats["next_state_entropy"]),
            "mean_next_state_radius": np.mean(ball_stats["next_state_radius"]),
            "fraction_deterministic": np.mean(ball_stats["is_deterministic"]),
            "variance_ratio": np.mean(ball_stats["next_state_variance"]) / (radius**2)
            if radius > 0
            else 0,
            "expansion_factor": np.mean(ball_stats["next_state_radius"]) / radius
            if radius > 0
            else 0,
            "ball_stats": ball_stats,
        }

        return results

    def _estimate_entropy(self, points, n_bins=10):
        """Estimate entropy of a point distribution."""
        if len(points) < 2:
            return 0

        # Simple entropy estimation by discretization
        n_dim = points.shape[1]
        entropies = []

        for dim in range(min(n_dim, 3)):  # Limit dimensions for efficiency
            hist, _ = np.histogram(points[:, dim], bins=n_bins)
            prob = hist / len(points)
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            entropies.append(entropy)

        return np.mean(entropies)

    def scan_radii(self, radii=None, min_points=5):
        """
        Scan across different radii to see how determinism changes.

        Args:
            radii: List of radii to test (if None, automatically generated)
            min_points: Minimum points per ball

        Returns:
            dict: Results for each radius
        """
        if radii is None:
            # Generate logarithmic scale of radii
            state_span = np.max(self.states, axis=0) - np.min(self.states, axis=0)
            mean_span = np.mean(state_span)
            radii = np.logspace(-2, 0, 10) * mean_span

        results = {}
        for r in radii:
            results[r] = self.compute_ball_statistics(r, min_points_per_ball=min_points)

        return results

    def visualize_determinism_scan(self, results):
        """
        Visualize how determinism metrics change with radius.
        """
        radii = list(results.keys())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Fraction of deterministic balls vs radius
        ax1 = axes[0, 0]
        frac_det = [results[r]["fraction_deterministic"] for r in radii]
        ax1.semilogx(radii, frac_det, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Ball Radius")
        ax1.set_ylabel("Fraction of Deterministic Balls")
        ax1.set_title("Determinism vs State Aggregation")
        ax1.grid(True, alpha=0.3)

        # 2. Next state variance vs radius
        ax2 = axes[0, 1]
        mean_var = [results[r]["mean_next_state_variance"] for r in radii]
        std_var = [results[r]["std_next_state_variance"] for r in radii]
        ax2.errorbar(
            radii, mean_var, yerr=std_var, fmt="rs-", linewidth=2, markersize=8
        )
        ax2.set_xlabel("Ball Radius")
        ax2.set_ylabel("Mean Next State Variance")
        ax2.set_title("Transition Variance vs Aggregation")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # 3. Expansion factor (how much next state ball expands/shrinks)
        ax3 = axes[0, 2]
        expansion = [results[r]["expansion_factor"] for r in radii]
        ax3.semilogx(radii, expansion, "gs-", linewidth=2, markersize=8)
        ax3.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="No expansion")
        ax3.set_xlabel("Ball Radius")
        ax3.set_ylabel("Expansion Factor")
        ax3.set_title("State Ball Expansion")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Next state entropy vs radius
        ax4 = axes[1, 0]
        entropy = [results[r]["mean_next_state_entropy"] for r in radii]
        ax4.semilogx(radii, entropy, "mo-", linewidth=2, markersize=8)
        ax4.set_xlabel("Ball Radius")
        ax4.set_ylabel("Mean Next State Entropy (bits)")
        ax4.set_title("Transition Entropy vs Aggregation")
        ax4.grid(True, alpha=0.3)

        # 5. Variance ratio (variance/radius^2)
        ax5 = axes[1, 1]
        var_ratio = [results[r]["variance_ratio"] for r in radii]
        ax5.semilogx(radii, var_ratio, "co-", linewidth=2, markersize=8)
        ax5.set_xlabel("Ball Radius")
        ax5.set_ylabel("Variance / Radius²")
        ax5.set_title("Normalized Transition Variance")
        ax5.grid(True, alpha=0.3)

        # 6. Points per ball
        ax6 = axes[1, 2]
        points_per_ball = [results[r]["mean_points_per_ball"] for r in radii]
        ax6.loglog(radii, points_per_ball, "k*-", linewidth=2, markersize=8)
        ax6.set_xlabel("Ball Radius")
        ax6.set_ylabel("Mean Points per Ball")
        ax6.set_title("Sampling Density")
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f"MDP Determinism Analysis - {self.env.spec.id}", fontsize=16)
        plt.tight_layout()
        plt.savefig("mdp_determinism_scan.png", dpi=150, bbox_inches="tight")
        plt.show()

    def visualize_ball_example(self, radius, ball_index=0):
        """
        Visualize a specific ball and its transitions.
        """
        results = self.compute_ball_statistics(radius)

        if results["n_balls"] == 0:
            print("No balls found with sufficient points")
            return

        # Get a specific ball
        ball_idx = min(ball_index, results["n_balls"] - 1)
        indices = results["ball_stats"]["state_indices"][ball_idx]

        # Get states in ball and their next states
        states_in_ball = self.states[indices]
        next_states_in_ball = self.next_states[indices]

        if self.state_dim >= 2:
            fig = plt.figure(figsize=(15, 5))

            # 2D or 3D visualization
            if self.state_dim == 2:
                # 2D plot
                ax1 = fig.add_subplot(131)
                ax1.scatter(
                    states_in_ball[:, 0],
                    states_in_ball[:, 1],
                    c="blue",
                    alpha=0.6,
                    label="States in ball",
                    s=50,
                )
                ax1.scatter(
                    next_states_in_ball[:, 0],
                    next_states_in_ball[:, 1],
                    c="red",
                    alpha=0.6,
                    label="Next states",
                    s=50,
                )

                # Draw arrows showing transitions
                for i in range(min(10, len(states_in_ball))):
                    ax1.arrow(
                        states_in_ball[i, 0],
                        states_in_ball[i, 1],
                        next_states_in_ball[i, 0] - states_in_ball[i, 0],
                        next_states_in_ball[i, 1] - states_in_ball[i, 1],
                        head_width=0.05,
                        head_length=0.05,
                        fc="gray",
                        ec="gray",
                        alpha=0.3,
                    )

                ax1.set_xlabel("State dim 1")
                ax1.set_ylabel("State dim 2")

            else:
                # 3D plot (first 3 dimensions)
                ax1 = fig.add_subplot(131, projection="3d")
                ax1.scatter(
                    states_in_ball[:, 0],
                    states_in_ball[:, 1],
                    states_in_ball[:, 2],
                    c="blue",
                    alpha=0.6,
                    label="States in ball",
                    s=50,
                )
                ax1.scatter(
                    next_states_in_ball[:, 0],
                    next_states_in_ball[:, 1],
                    next_states_in_ball[:, 2],
                    c="red",
                    alpha=0.6,
                    label="Next states",
                    s=50,
                )
                ax1.set_xlabel("Dim 1")
                ax1.set_ylabel("Dim 2")
                ax1.set_zlabel("Dim 3")

            ax1.set_title(f"Ball (r={radius}) and Transitions")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Distribution of next state distances
            ax2 = fig.add_subplot(132)
            center_state = np.mean(states_in_ball, axis=0)
            next_center = np.mean(next_states_in_ball, axis=0)

            distances_from_center = np.linalg.norm(
                next_states_in_ball - next_center, axis=1
            )
            ax2.hist(
                distances_from_center,
                bins=20,
                alpha=0.7,
                color="red",
                edgecolor="black",
            )
            ax2.axvline(
                np.mean(distances_from_center),
                color="k",
                linestyle="--",
                label=f"Mean: {np.mean(distances_from_center):.3f}",
            )
            ax2.set_xlabel("Distance from Next State Center")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Next State Spread")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Transition vectors
            ax3 = fig.add_subplot(133)
            vectors = next_states_in_ball - states_in_ball
            if self.state_dim >= 2:
                ax3.quiver(
                    states_in_ball[:20, 0],
                    states_in_ball[:20, 1],
                    vectors[:20, 0],
                    vectors[:20, 1],
                    alpha=0.7,
                    scale=1,
                    scale_units="xy",
                )
            ax3.set_xlabel("State dim 1")
            ax3.set_ylabel("State dim 2")
            ax3.set_title("Transition Vectors")
            ax3.grid(True, alpha=0.3)

            plt.suptitle(
                f"Ball Analysis - Variance: {results['mean_next_state_variance']:.4f}",
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig("ball_example.png", dpi=150, bbox_inches="tight")
            plt.show()

        # Print statistics
        print(f"\nBall Statistics (radius = {radius}):")
        print(f"  Number of points: {len(indices)}")
        print(f"  Next state variance: {results['mean_next_state_variance']:.6f}")
        print(f"  Next state entropy: {results['mean_next_state_entropy']:.3f} bits")
        print(f"  Expansion factor: {results['expansion_factor']:.3f}")
        print(f"  Is deterministic: {results['fraction_deterministic']:.2%}")

    def compute_scaling_law(self, results):
        """
        Compute how determinism scales with radius.
        """
        radii = np.array(list(results.keys()))
        frac_det = np.array([results[r]["fraction_deterministic"] for r in radii])
        variance = np.array([results[r]["mean_next_state_variance"] for r in radii])

        # Fit power law: variance ~ radius^alpha
        log_r = np.log(radii[radii > 0])
        log_var = np.log(variance[radii > 0])

        if len(log_r) > 1:
            coeffs = np.polyfit(log_r, log_var, 1)
            alpha = coeffs[0]
            print(f"\nScaling Law Analysis:")
            print(f"  Variance ∝ radius^{alpha:.2f}")

            if alpha < 1:
                print("  → Transitions are CONTRACTIVE (errors shrink)")
            elif alpha > 2:
                print("  → Transitions are EXPANSIVE (errors amplify)")
            else:
                print("  → Transitions are NEUTRALLY SCALING")

        # Find critical radius where determinism drops below 50%
        critical_r = None
        for i, r in enumerate(radii):
            if frac_det[i] < 0.5:
                critical_r = r
                break

        if critical_r is not None:
            print(f"  Critical radius (50% determinism): {critical_r:.4f}")

        return {
            "scaling_exponent": alpha if len(log_r) > 1 else None,
            "critical_radius": critical_r,
        }


# Run experiments
def run_determinism_experiment(env_id, use_trained_policy=True, n_samples=5000):
    """
    Run complete determinism validation experiment.
    """
    print("=" * 60)
    print(f"MDP Determinism Validation - {env_id}")
    print("=" * 60)

    # Create environment
    env = gym.make(env_id)

    # Train policy if requested
    if use_trained_policy:
        print("\nTraining SAC policy...")
        model = SAC("MlpPolicy", env)
        # model.load("sac_pendulum")
        model.learn(total_timesteps=30000)
        policy = model.policy
    else:
        policy = None

    # Create validator
    validator = MDPDeterminismValidator(env, policy, n_samples=n_samples)

    # Collect transitions
    validator.collect_transitions()

    # Scan radii
    print("\nScanning across radii...")
    results = validator.scan_radii(min_points=5)

    # Visualize results
    validator.visualize_determinism_scan(results)

    # Show example ball
    median_r = np.median(list(results.keys()))
    validator.visualize_ball_example(median_r, ball_index=0)

    # Compute scaling law
    scaling = validator.compute_scaling_law(results)

    # Comparative analysis: random vs trained policy
    if use_trained_policy:
        print("\n" + "=" * 60)
        print("Comparing with Random Policy")
        print("=" * 60)

        random_validator = MDPDeterminismValidator(env, None, n_samples=n_samples // 2)
        random_validator.collect_transitions()
        random_results = random_validator.scan_radii(min_points=5)

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        radii = list(results.keys())
        frac_det_trained = [results[r]["fraction_deterministic"] for r in radii]
        frac_det_random = [
            random_results[r]["fraction_deterministic"]
            for r in radii
            if r in random_results
        ]

        axes[0].semilogx(radii, frac_det_trained, "b-", linewidth=2, label="Trained")
        axes[0].semilogx(
            list(random_results.keys()),
            frac_det_random,
            "r--",
            linewidth=2,
            label="Random",
        )
        axes[0].set_xlabel("Ball Radius")
        axes[0].set_ylabel("Fraction Deterministic")
        axes[0].set_title("Determinism: Trained vs Random")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        var_trained = [results[r]["mean_next_state_variance"] for r in radii]
        var_random = [
            random_results[r]["mean_next_state_variance"]
            for r in radii
            if r in random_results
        ]

        axes[1].loglog(radii, var_trained, "b-", linewidth=2, label="Trained")
        axes[1].loglog(
            list(random_results.keys()), var_random, "r--", linewidth=2, label="Random"
        )
        axes[1].set_xlabel("Ball Radius")
        axes[1].set_ylabel("Mean Next State Variance")
        axes[1].set_title("Transition Variance")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Policy Comparison - {env_id}")
        plt.tight_layout()
        plt.savefig("policy_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

    env.close()
    return validator, results


if __name__ == "__main__":
    # Test on different environments
    env_ids = ["MountainCarContinuous-v0"]

    for env_id in env_ids:
        try:
            validator, results = run_determinism_experiment(
                env_id, use_trained_policy=True, n_samples=3000
            )

            # Print summary
            print("\n" + "=" * 60)
            print(f"SUMMARY - {env_id}")
            print("=" * 60)

            for r, res in results.items():
                print(f"\nRadius: {r:.4f}")
                print(f"  Determinism: {res['fraction_deterministic']:.2%}")
                print(f"  Points/ball: {res['mean_points_per_ball']:.1f}")
                print(f"  Variance: {res['mean_next_state_variance']:.6f}")
                print(f"  Entropy: {res['mean_next_state_entropy']:.3f} bits")

        except Exception as e:
            print(f"Error with {env_id}: {e}")
            import traceback

            traceback.print_exc()
