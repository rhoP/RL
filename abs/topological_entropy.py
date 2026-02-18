import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings("ignore")


class LocalEntropyEstimator:
    """
    Estimate local entropy h_ϵ(s0) = limsup_{T→∞} log N_ϵ(s0,T)/T
    where N_ϵ(s0,T) is number of trajectories starting within ϵ-ball
    that remain distinguishable.
    """

    def __init__(self, env, policy, epsilon=0.01, device="cpu"):
        self.env = env
        self.policy = policy
        self.epsilon = epsilon
        self.device = device

    def get_nearby_states(self, state, K, method="uniform"):
        """Sample K nearby states within epsilon-ball of the given state."""
        nearby_states = []
        state = np.array(state).flatten()  # Ensure state is 1D

        if method == "uniform":
            # Uniform sampling within epsilon-ball
            for _ in range(K):
                perturbation = np.random.uniform(
                    -self.epsilon, self.epsilon, size=state.shape
                )
                new_state = state + perturbation
                # Clip to stay within valid bounds if necessary
                if hasattr(self.env, "observation_space") and hasattr(
                    self.env.observation_space, "low"
                ):
                    new_state = np.clip(
                        new_state,
                        self.env.observation_space.low,
                        self.env.observation_space.high,
                    )
                nearby_states.append(new_state)

        elif method == "gaussian":
            # Gaussian sampling with std = epsilon/3 (so 99.7% within epsilon)
            for _ in range(K):
                perturbation = np.random.normal(0, self.epsilon / 3, size=state.shape)
                new_state = state + perturbation
                if hasattr(self.env, "observation_space") and hasattr(
                    self.env.observation_space, "low"
                ):
                    new_state = np.clip(
                        new_state,
                        self.env.observation_space.low,
                        self.env.observation_space.high,
                    )
                nearby_states.append(new_state)

        return np.array(nearby_states)

    def compute_trajectory_distances(self, trajectories):
        """
        Compute pairwise distances between trajectories at each timestep.
        trajectories shape: (K, T, state_dim)
        """
        K, T, state_dim = trajectories.shape
        distances = np.zeros((T, K, K))

        for t in range(T):
            # Get states at time t for all trajectories
            states_t = trajectories[:, t, :]
            # Compute pairwise distances
            distances[t] = squareform(pdist(states_t, metric="euclidean"))

        return distances

    def count_distinguishable_trajectories(self, trajectories, threshold=None):
        """
        Count number of trajectories that remain distinguishable.
        Two trajectories are distinguishable if their distance > threshold.
        """
        if threshold is None:
            threshold = self.epsilon

        K, T, _ = trajectories.shape
        if K == 0 or T == 0:
            return np.array([0])

        distances = self.compute_trajectory_distances(trajectories)

        # Track distinguishable trajectories over time
        distinguishable_count = []

        for t in range(T):
            # Get distances at time t
            dist_t = distances[t]
            # Use a clustering approach to count distinct trajectories
            # Two trajectories are in the same cluster if their distance < threshold
            distinguishable = 0
            used = np.zeros(K, dtype=bool)

            for i in range(K):
                if not used[i]:
                    # Find all trajectories close to i
                    cluster = np.where(dist_t[i] < threshold)[0]
                    used[cluster] = True
                    distinguishable += 1

            distinguishable_count.append(distinguishable)

        return np.array(distinguishable_count)

    def estimate_local_entropy(self, s0, K=100, T_max=100, n_runs=5):
        """
        Estimate local entropy h_ϵ(s0) for a given starting state.
        """
        entropy_estimates = []
        all_counts = []

        for run in range(n_runs):
            # Sample K nearby states
            nearby_states = self.get_nearby_states(s0, K)

            # Generate trajectories for each starting state
            trajectories = []

            for state in nearby_states:
                trajectory = [state.copy()]

                # Create a fresh environment instance for each trajectory
                env_copy = gym.make(self.env.spec.id)

                # Reset and try to set initial state
                obs, _ = env_copy.reset()

                # Try to set the state directly if the environment supports it
                try:
                    if hasattr(env_copy.unwrapped, "state"):
                        env_copy.unwrapped.state = state.copy()
                    # For environments that use a different attribute name
                    elif hasattr(env_copy.unwrapped, "_state"):
                        env_copy.unwrapped._state = state.copy()
                except:
                    pass

                # Run trajectory with deterministic policy
                for t in range(T_max):
                    with torch.no_grad():
                        # Get deterministic action from policy
                        if hasattr(self.policy, "predict"):
                            action, _ = self.policy.predict(obs, deterministic=True)
                        else:
                            # For custom policies
                            obs_tensor = (
                                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                            )
                            action = self.policy(obs_tensor).cpu().numpy()[0]

                    # Take step in environment
                    obs, reward, terminated, truncated, info = env_copy.step(action)
                    trajectory.append(obs.copy())

                    if terminated or truncated:
                        # Pad trajectory if episode ends early
                        while len(trajectory) < T_max + 1:
                            trajectory.append(obs.copy())
                        break

                # Ensure trajectory has exactly T_max+1 steps
                if len(trajectory) < T_max + 1:
                    while len(trajectory) < T_max + 1:
                        trajectory.append(trajectory[-1].copy())
                elif len(trajectory) > T_max + 1:
                    trajectory = trajectory[: T_max + 1]

                trajectories.append(np.array(trajectory))
                env_copy.close()

            trajectories = np.array(trajectories)

            # Count distinguishable trajectories
            distinguishable_counts = self.count_distinguishable_trajectories(
                trajectories
            )
            all_counts.append(distinguishable_counts)

            # Estimate entropy as slope of log(N) vs T
            if len(distinguishable_counts) > 10 and np.min(distinguishable_counts) > 1:
                log_counts = np.log(distinguishable_counts)

                # Fit linear regression for the latter part of the trajectory
                # Use at least 10 points for fitting
                T_mid = min(
                    len(distinguishable_counts) // 3, len(distinguishable_counts) - 10
                )
                T_mid = max(T_mid, 5)  # Ensure we have enough points

                if len(distinguishable_counts) - T_mid >= 5:
                    x_data = np.arange(T_mid, len(distinguishable_counts))
                    y_data = log_counts[T_mid:]

                    # Ensure same length
                    min_len = min(len(x_data), len(y_data))
                    x_data = x_data[:min_len]
                    y_data = y_data[:min_len]

                    if len(x_data) >= 5:
                        coeffs = np.polyfit(x_data, y_data, 1)
                        entropy_estimates.append(
                            max(0, coeffs[0])
                        )  # Entropy should be non-negative
                    else:
                        entropy_estimates.append(0)
                else:
                    entropy_estimates.append(0)
            else:
                entropy_estimates.append(0)

        # If all estimates are zero, return zero with appropriate structure
        if len(entropy_estimates) == 0 or all(e == 0 for e in entropy_estimates):
            dummy_counts = np.ones(T_max + 1) * K
            return {
                "entropy_mean": 0.0,
                "entropy_std": 0.0,
                "counts_mean": dummy_counts,
                "counts_std": np.zeros_like(dummy_counts),
                "individual_estimates": [0.0] * n_runs,
            }

        # Ensure all counts arrays have the same length
        max_len = max(len(counts) for counts in all_counts)
        padded_counts = []
        for counts in all_counts:
            if len(counts) < max_len:
                padded = np.pad(counts, (0, max_len - len(counts)), "edge")
                padded_counts.append(padded)
            else:
                padded_counts.append(counts[:max_len])

        counts_mean = np.mean(padded_counts, axis=0)
        counts_std = np.std(padded_counts, axis=0)

        return {
            "entropy_mean": np.mean(entropy_estimates),
            "entropy_std": np.std(entropy_estimates),
            "counts_mean": counts_mean,
            "counts_std": counts_std,
            "individual_estimates": entropy_estimates,
        }


def train_policy(env_id, algorithm="SAC", total_timesteps=20000):
    """Train a policy in the given environment."""
    env = gym.make(env_id)

    if algorithm == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)
    elif algorithm == "TD3":
        model = TD3("MlpPolicy", env, verbose=1)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Training {algorithm} on {env_id} for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    return model, env


def run_experiment(
    env_ids, algorithms, epsilon=0.01, K=20, T_max=30, n_start_states=2, n_runs=2
):
    """
    Run complete experiment across multiple environments and algorithms.
    """
    results = {}

    for env_id in env_ids:
        results[env_id] = {}
        for algo in algorithms:
            print(f"\n{'=' * 60}")
            print(f"Running {algo} on {env_id}")
            print("=" * 60)

            try:
                # Train policy
                model, env = train_policy(env_id, algo)

                # Create entropy estimator
                estimator = LocalEntropyEstimator(env, model.policy, epsilon=epsilon)

                # Sample starting states from the environment
                start_states = []
                for _ in range(n_start_states):
                    obs, _ = env.reset()
                    start_states.append(obs.copy())

                # Estimate local entropy for each starting state
                env_results = []
                for i, s0 in enumerate(start_states):
                    print(f"Starting state {i + 1}/{n_start_states}")
                    entropy_data = estimator.estimate_local_entropy(
                        s0, K=K, T_max=T_max, n_runs=n_runs
                    )
                    env_results.append(
                        {"start_state": s0, "entropy_data": entropy_data}
                    )

                results[env_id][algo] = env_results

                # Clean up
                env.close()

            except Exception as e:
                print(f"Error running {algo} on {env_id}: {e}")
                results[env_id][algo] = []

    return results


def plot_results(results, epsilon, save_path=None):
    """Plot the experimental results."""
    # Filter out empty results
    non_empty_results = {
        env: {algo: res for algo, res in algos.items() if res}
        for env, algos in results.items()
    }
    non_empty_results = {
        env: algos for env, algos in non_empty_results.items() if algos
    }

    if not non_empty_results:
        print("No results to plot")
        return

    n_envs = len(non_empty_results)
    n_algos = max(len(algos) for algos in non_empty_results.values())

    fig, axes = plt.subplots(n_envs, n_algos, figsize=(5 * n_algos, 4 * n_envs))
    if n_envs == 1:
        axes = axes.reshape(1, -1)
    if n_algos == 1:
        axes = axes.reshape(-1, 1)

    for i, (env_id, env_results) in enumerate(non_empty_results.items()):
        for j, (algo, algo_results) in enumerate(env_results.items()):
            ax = axes[i, j]

            if algo_results:
                # Plot entropy estimates
                entropies = [r["entropy_data"]["entropy_mean"] for r in algo_results]
                entropies_std = [r["entropy_data"]["entropy_std"] for r in algo_results]

                x = np.arange(len(entropies))
                ax.bar(x, entropies, yerr=entropies_std, capsize=5, alpha=0.7)
                ax.set_xlabel("Starting State Index")
                ax.set_ylabel(f"Local Entropy h_ε (ε={epsilon})")
                ax.set_title(f"{env_id} - {algo}")
                ax.grid(True, alpha=0.3)

                # Add mean line
                if entropies:
                    mean_entropy = np.mean(entropies)
                    ax.axhline(
                        y=mean_entropy,
                        color="r",
                        linestyle="--",
                        label=f"Mean: {mean_entropy:.3f}",
                    )
                    ax.legend()
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def analyze_trajectory_divergence(results, env_id, algo, start_state_idx=0):
    """Analyze how trajectories diverge over time for a specific case."""
    if (
        env_id not in results
        or algo not in results[env_id]
        or not results[env_id][algo]
    ):
        print(f"No data available for {env_id} - {algo}")
        return

    env_results = results[env_id][algo][start_state_idx]
    counts_mean = env_results["entropy_data"]["counts_mean"]
    counts_std = env_results["entropy_data"]["counts_std"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot distinguishable trajectories over time
    T = len(counts_mean)
    ax1.errorbar(range(T), counts_mean, yerr=counts_std, capsize=3)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Number of Distinguishable Trajectories")
    ax1.set_title(f"Trajectory Divergence - {env_id} - {algo}")
    ax1.grid(True, alpha=0.3)

    # Plot log(N) vs time
    log_counts = np.log(np.maximum(counts_mean, 1))
    ax2.plot(range(T), log_counts)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("log(Number of Distinguishable Trajectories)")
    ax2.set_title("Log-Linear Plot (slope = local entropy)")
    ax2.grid(True, alpha=0.3)

    # Fit line to estimate entropy
    if T > 10:
        T_mid = T // 3
        if T - T_mid >= 5:
            x_data = np.arange(T_mid, T)
            y_data = log_counts[T_mid:]
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]

            if len(x_data) >= 5:
                coeffs = np.polyfit(x_data, y_data, 1)
                ax2.plot(
                    x_data,
                    np.polyval(coeffs, x_data),
                    "r--",
                    label=f"Slope: {coeffs[0]:.3f}",
                )
                ax2.legend()

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Configuration
    ENV_IDS = [
        "LunarLanderContinuous-v3",
        "MountainCarContinuous-v0",
    ]

    ALGORITHMS = ["SAC", "TD3", "PPO"]
    EPSILON = 0.01
    K = 15  # Number of nearby trajectories
    T_MAX = 25  # Maximum trajectory length
    N_START_STATES = 2  # Number of starting states to test
    N_RUNS = 2  # Number of runs for averaging

    print("Starting local entropy experiment...")
    print(f"Environments: {ENV_IDS}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Parameters: ε={EPSILON}, K={K}, T_max={T_MAX}")

    # Run experiment
    results = run_experiment(
        env_ids=ENV_IDS,
        algorithms=ALGORITHMS,
        epsilon=EPSILON,
        K=K,
        T_max=T_MAX,
        n_start_states=N_START_STATES,
        n_runs=N_RUNS,
    )

    # Plot summary results
    plot_results(results, EPSILON, save_path="local_entropy_results.png")

    # Print numerical results
    print("\n" + "=" * 60)
    print("NUMERICAL RESULTS")
    print("=" * 60)

    for env_id in ENV_IDS:
        print(f"\nEnvironment: {env_id}")
        for algo in ALGORITHMS:
            if env_id in results and algo in results[env_id] and results[env_id][algo]:
                entropies = [
                    r["entropy_data"]["entropy_mean"] for r in results[env_id][algo]
                ]
                if entropies:
                    print(
                        f"  {algo}: mean entropy = {np.mean(entropies):.4f} ± {np.std(entropies):.4f}"
                    )
                else:
                    print(f"  {algo}: No data available")
            else:
                print(f"  {algo}: No data available")

    # Detailed analysis for a specific case if available
    if (
        "Pendulum-v1" in results
        and "SAC" in results["Pendulum-v1"]
        and results["Pendulum-v1"]["SAC"]
    ):
        analyze_trajectory_divergence(results, "Pendulum-v1", "SAC", start_state_idx=0)
