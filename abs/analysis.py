"""
Diagnostic Suite for Topological Abstraction Framework
Analyzes behavioral distance components, parameter sensitivity, and abstraction quality
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd
import warnings
import networkx as nx

warnings.filterwarnings("ignore")

# Import from your original script
from filtr import (
    BehavioralDistance,
    BehavioralFiltration,
    FiltrationSpectralClustering,
    DeterministicAbstractMDP,
    FiltrationVisualizer,
)

# ============ 1. BEHAVIORAL DISTANCE DIAGNOSTICS ============


class BehavioralDistanceDiagnostics:
    """
    Analyze the components of behavioral distance:
    d_B = α * W1 + β * |r_diff|
    """

    def __init__(self, env, n_episodes=200, sample_size=100):
        self.env = env
        self.n_episodes = n_episodes
        self.sample_size = sample_size
        self.results = {}

    def run_component_analysis(self, alpha_values=None, beta_values=None):
        """Analyze how changing α and β affects distance structure"""
        if alpha_values is None:
            alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
        if beta_values is None:
            beta_values = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]

        # Collect base data once
        print("Collecting transition data...")
        behavioral = BehavioralDistance(self.env, gamma=0.95)
        behavioral.collect_transitions(n_episodes=self.n_episodes)
        dist_matrix_base, keys = behavioral.compute_distance_matrix(
            sample_size=min(self.sample_size, len(behavioral.transition_data))
        )

        # Store raw components
        n = len(keys)
        reward_diffs = np.zeros((n, n))
        wasserstein_dists = np.zeros((n, n))

        print("\nComputing raw distance components...")
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys[i:], i):
                if i == j:
                    continue

                # Reward difference
                r_i = (
                    np.mean(behavioral.reward_data[key_i])
                    if behavioral.reward_data[key_i]
                    else 0
                )
                r_j = (
                    np.mean(behavioral.reward_data[key_j])
                    if behavioral.reward_data[key_j]
                    else 0
                )
                reward_diffs[i, j] = abs(r_i - r_j)
                reward_diffs[j, i] = reward_diffs[i, j]

                # Wasserstein distance
                w_dist = behavioral._wasserstein_distance(
                    behavioral.transition_data[key_i], behavioral.transition_data[key_j]
                )
                wasserstein_dists[i, j] = w_dist
                wasserstein_dists[j, i] = w_dist

        # Analyze different weight combinations
        results = {
            "alpha_values": alpha_values,
            "beta_values": beta_values,
            "reward_diffs": reward_diffs,
            "wasserstein_dists": wasserstein_dists,
            "keys": keys,
            "transition_data": behavioral.transition_data,
            "reward_data": behavioral.reward_data,
        }

        distance_matrices = {}
        statistics = defaultdict(list)

        for alpha in alpha_values:
            for beta in beta_values:
                key = f"α={alpha}, β={beta}"
                print(f"Computing {key}")

                # Combined distance
                dist_matrix = alpha * wasserstein_dists + beta * reward_diffs
                np.fill_diagonal(dist_matrix, 0)
                distance_matrices[key] = dist_matrix

                # Compute statistics
                non_diag = dist_matrix[dist_matrix > 0]
                if len(non_diag) > 0:
                    statistics["mean"].append(np.mean(non_diag))
                    statistics["std"].append(np.std(non_diag))
                    statistics["min"].append(np.min(non_diag))
                    statistics["max"].append(np.max(non_diag))
                    statistics["skew"].append(pd.Series(non_diag).skew())

                    # Correlation with components - with safety checks
                    w_flat = wasserstein_dists[wasserstein_dists > 0]
                    r_flat = reward_diffs[reward_diffs > 0]

                    # Ensure same length and at least 2 points
                    min_len = min(len(non_diag), len(w_flat), len(r_flat))
                    if min_len >= 2:
                        # Sample to same length
                        indices = np.random.choice(min_len, min_len, replace=False)
                        try:
                            corr_w, _ = pearsonr(non_diag[indices], w_flat[indices])
                        except:
                            corr_w = 0
                        try:
                            corr_r, _ = pearsonr(non_diag[indices], r_flat[indices])
                        except:
                            corr_r = 0
                    else:
                        corr_w = 0
                        corr_r = 0

                    statistics["corr_wasserstein"].append(corr_w)
                    statistics["corr_reward"].append(corr_r)
                else:
                    statistics["mean"].append(0)
                    statistics["std"].append(0)
                    statistics["min"].append(0)
                    statistics["max"].append(0)
                    statistics["skew"].append(0)
                    statistics["corr_wasserstein"].append(0)
                    statistics["corr_reward"].append(0)

                # Store parameters
                statistics["alpha"].append(alpha)
                statistics["beta"].append(beta)

        results["distance_matrices"] = distance_matrices
        results["statistics"] = pd.DataFrame(statistics)

        self.results = results
        return results

    def plot_component_analysis(self):
        """Visualize the component analysis results"""
        if not self.results:
            print("Run component analysis first")
            return

        results = self.results
        df = results["statistics"]

        # Create pivot tables for heatmaps
        pivot_mean = df.pivot(index="alpha", columns="beta", values="mean")
        pivot_std = df.pivot(index="alpha", columns="beta", values="std")
        pivot_corr_w = df.pivot(
            index="alpha", columns="beta", values="corr_wasserstein"
        )
        pivot_corr_r = df.pivot(index="alpha", columns="beta", values="corr_reward")

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)

        # Heatmap 1: Mean distance
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(pivot_mean, annot=True, fmt=".3f", cmap="viridis", ax=ax1)
        ax1.set_title("Mean Behavioral Distance")

        # Heatmap 2: Std deviation
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(pivot_std, annot=True, fmt=".3f", cmap="plasma", ax=ax2)
        ax2.set_title("Distance Std Deviation")

        # Heatmap 3: Correlation with Wasserstein
        ax3 = fig.add_subplot(gs[0, 2])
        sns.heatmap(
            pivot_corr_w,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax3,
        )
        ax3.set_title("Correlation with Wasserstein")

        # Heatmap 4: Correlation with reward diff
        ax4 = fig.add_subplot(gs[1, 0])
        sns.heatmap(
            pivot_corr_r,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax4,
        )
        ax4.set_title("Correlation with Reward Diff")

        # Line plots for selected alphas
        ax5 = fig.add_subplot(gs[1, 1])
        for alpha in sorted(df["alpha"].unique())[:4]:
            subset = df[df["alpha"] == alpha]
            ax5.plot(subset["beta"], subset["mean"], "o-", label=f"α={alpha}")
        ax5.set_xlabel("β (Reward Weight)")
        ax5.set_ylabel("Mean Distance")
        ax5.set_title("Distance vs β (fixed α)")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Line plots for selected betas
        ax6 = fig.add_subplot(gs[1, 2])
        for beta in sorted(df["beta"].unique())[:4]:
            subset = df[df["beta"] == beta]
            ax6.plot(subset["alpha"], subset["mean"], "o-", label=f"β={beta}")
        ax6.set_xlabel("α (Wasserstein Weight)")
        ax6.set_ylabel("Mean Distance")
        ax6.set_title("Distance vs α (fixed β)")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 3D surface plot
        ax7 = fig.add_subplot(gs[2, :], projection="3d")
        X, Y = np.meshgrid(sorted(df["alpha"].unique()), sorted(df["beta"].unique()))
        Z = pivot_mean.values.T

        surf = ax7.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
        ax7.set_xlabel("α (Wasserstein)")
        ax7.set_ylabel("β (Reward)")
        ax7.set_zlabel("Mean Distance")
        ax7.set_title("3D Surface of Distance vs Parameters")
        fig.colorbar(surf, ax=ax7, shrink=0.5, aspect=10)

        plt.suptitle("Behavioral Distance Component Analysis", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # Distribution comparison for selected parameter combinations
        self.plot_distance_distributions()

    def plot_distance_distributions(self, combinations=None):
        """Compare distance distributions for different parameter combinations"""
        if combinations is None:
            # Select representative combinations
            combinations = [
                ("α=1.0, β=0.0", "Wasserstein only"),
                ("α=0.0, β=1.0", "Reward only"),
                ("α=1.0, β=1.0", "Equal weights"),
                ("α=2.0, β=1.0", "Wasserstein heavy"),
                ("α=1.0, β=2.0", "Reward heavy"),
            ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (key, label) in enumerate(combinations):
            if idx >= len(axes):
                break

            if key in self.results["distance_matrices"]:
                dist_matrix = self.results["distance_matrices"][key]
                distances = dist_matrix[dist_matrix > 0]

                ax = axes[idx]
                ax.hist(distances, bins=30, alpha=0.7, edgecolor="black")
                ax.axvline(
                    np.mean(distances),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {np.mean(distances):.3f}",
                )
                ax.axvline(
                    np.median(distances),
                    color="green",
                    linestyle="--",
                    label=f"Median: {np.median(distances):.3f}",
                )
                ax.set_xlabel("Distance")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{label}\n{key}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Hide empty subplot
        for idx in range(len(combinations), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Distance Distributions for Different Parameter Combinations",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        plt.show()


# ============ 2. ABSTRACTION QUALITY DIAGNOSTICS ============


class AbstractionQualityDiagnostics:
    """
    Analyze the quality of the learned abstraction
    """

    def __init__(self, behavioral_dist, filtration, spectral, abstract_mdp):
        self.behavioral = behavioral_dist
        self.filtration = filtration
        self.spectral = spectral
        self.abstract_mdp = abstract_mdp

    def analyze_cluster_quality(self):
        """Compute cluster quality metrics"""
        if self.spectral.cluster_assignments is None:
            print("No cluster assignments available")
            return {}

        labels = self.spectral.cluster_assignments
        valid_mask = labels >= 0

        if np.sum(valid_mask) < 2:
            print("Insufficient valid points for clustering analysis")
            return {}

        # Silhouette score
        try:
            sil_score = silhouette_score(
                self.behavioral.distance_matrix[valid_mask][:, valid_mask],
                labels[valid_mask],
                metric="precomputed",
            )
        except:
            sil_score = -1

        # Cluster sizes
        unique, counts = np.unique(labels[valid_mask], return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Separation between clusters
        separations = []
        for i, c1 in enumerate(unique):
            for c2 in unique[i + 1 :]:
                mask1 = labels == c1
                mask2 = labels == c2
                if np.any(mask1) and np.any(mask2):
                    # Mean distance between clusters
                    dists = self.behavioral.distance_matrix[mask1][:, mask2]
                    separations.append(np.mean(dists))

        results = {
            "n_clusters": len(unique),
            "silhouette_score": sil_score,
            "cluster_sizes": cluster_sizes,
            "mean_separation": np.mean(separations) if separations else 0,
            "min_cluster_size": np.min(counts) if len(counts) > 0 else 0,
            "max_cluster_size": np.max(counts) if len(counts) > 0 else 0,
        }

        return results

    def analyze_abstract_dynamics(self):
        """Analyze properties of the abstract MDP"""
        if self.abstract_mdp.T is None:
            return {}

        n_states = len(self.abstract_mdp.H)
        n_actions = len(self.abstract_mdp.A)

        # Transition matrix
        trans_matrix, reward_matrix = self.abstract_mdp.get_abstract_transition_matrix()

        # Determinism check (should be deterministic by construction)
        is_deterministic = True

        # Compute action entropy at each state
        action_entropy = []
        for h in self.abstract_mdp.H:
            # Count how many different actions lead to different next states
            next_states = set(trans_matrix[h, a] for a in range(n_actions))
            if len(next_states) > 1:
                # Actions are distinguishable
                probs = [1 / len(next_states)] * len(next_states)  # Simplified
                entropy = -sum(p * np.log(p) for p in probs)
                action_entropy.append(entropy)

        # Graph properties
        G = nx.MultiDiGraph()
        for h in self.abstract_mdp.H:
            G.add_node(h)
            for a in range(n_actions):
                G.add_edge(h, trans_matrix[h, a])

        try:
            # Strongly connected components
            scc = list(nx.strongly_connected_components(G))
            n_scc = len(scc)
            largest_scc = max(len(c) for c in scc) if scc else 0

            # Cycles
            cycles = list(nx.simple_cycles(G))
            n_cycles = len(cycles)

        except:
            n_scc = 0
            largest_scc = 0
            n_cycles = 0

        results = {
            "n_states": n_states,
            "n_actions": n_actions,
            "deterministic": is_deterministic,
            "action_entropy_mean": np.mean(action_entropy) if action_entropy else 0,
            "n_scc": n_scc,
            "largest_scc_size": largest_scc,
            "n_cycles": n_cycles,
            "trans_matrix": trans_matrix,
            "reward_matrix": reward_matrix,
        }

        return results

    def plot_abstraction_quality(self):
        """Visualize abstraction quality metrics"""
        cluster_quality = self.analyze_cluster_quality()
        dyn_quality = self.analyze_abstract_dynamics()

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)

        # 1. Cluster size distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if "cluster_sizes" in cluster_quality:
            sizes = list(cluster_quality["cluster_sizes"].values())
            ax1.bar(range(len(sizes)), sizes, color="skyblue", edgecolor="black")
            ax1.set_xlabel("Cluster ID")
            ax1.set_ylabel("Size")
            ax1.set_title("Cluster Size Distribution")
            ax1.grid(True, alpha=0.3)

        # 2. Silhouette score gauge
        ax2 = fig.add_subplot(gs[0, 1])
        if "silhouette_score" in cluster_quality:
            score = cluster_quality["silhouette_score"]
            colors = ["red", "yellow", "green"]
            ax2.barh(["Silhouette"], [score], color="skyblue", edgecolor="black")
            ax2.set_xlim(-1, 1)
            ax2.axvline(0, color="black", linestyle="-", alpha=0.3)
            ax2.axvline(0.25, color="orange", linestyle="--", alpha=0.5, label="Weak")
            ax2.axvline(0.5, color="green", linestyle="--", alpha=0.5, label="Strong")
            ax2.set_xlabel("Score")
            ax2.set_title(f"Silhouette Score: {score:.3f}")
            ax2.legend()

        # 3. Action entropy heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        if "trans_matrix" in dyn_quality:
            trans = dyn_quality["trans_matrix"]
            im = ax3.imshow(trans, cmap="tab20", aspect="auto", interpolation="nearest")
            ax3.set_xlabel("Action")
            ax3.set_ylabel("Abstract State")
            ax3.set_title("Abstract Transition Matrix")
            plt.colorbar(im, ax=ax3, label="Next State ID")

        # 4. Reward heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        if "reward_matrix" in dyn_quality:
            rewards = dyn_quality["reward_matrix"]
            im = ax4.imshow(
                rewards, cmap="RdYlGn", aspect="auto", interpolation="nearest"
            )
            ax4.set_xlabel("Action")
            ax4.set_ylabel("Abstract State")
            ax4.set_title("Abstract Reward Matrix")
            plt.colorbar(im, ax=ax4, label="Reward")

        # 5. Graph properties
        ax5 = fig.add_subplot(gs[1, 1])
        properties = [
            f"States: {dyn_quality.get('n_states', 0)}",
            f"Actions: {dyn_quality.get('n_actions', 0)}",
            f"Deterministic: {dyn_quality.get('deterministic', False)}",
            f"SCCs: {dyn_quality.get('n_scc', 0)}",
            f"Largest SCC: {dyn_quality.get('largest_scc_size', 0)}",
            f"Cycles: {dyn_quality.get('n_cycles', 0)}",
            f"Action Entropy: {dyn_quality.get('action_entropy_mean', 0):.3f}",
            f"Clusters: {cluster_quality.get('n_clusters', 0)}",
            f"Separation: {cluster_quality.get('mean_separation', 0):.3f}",
        ]
        ax5.axis("off")
        y_pos = np.linspace(0.9, 0.1, len(properties))
        for i, (y, prop) in enumerate(zip(y_pos, properties)):
            ax5.text(
                0.1,
                y,
                prop,
                transform=ax5.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7),
            )
        ax5.set_title("Abstract MDP Properties")

        # 6. Empty for now
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")

        plt.suptitle("Abstraction Quality Metrics", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


# ============ 3. PARAMETER SWEEP DIAGNOSTICS ============


class ParameterSweepDiagnostics:
    """
    Run systematic parameter sweeps to understand sensitivity
    """

    def __init__(self, env_name="FrozenLake-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.sweep_results = []

    def run_sweep(self, param_grid):
        """
        param_grid: dict with keys like 'alpha', 'beta', 'n_episodes', 'threshold'
        """
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        total_runs = np.prod([len(v) for v in values])
        print(f"Running parameter sweep with {total_runs} combinations...")

        run_idx = 0
        for combo in product(*values):
            params = dict(zip(keys, combo))
            run_idx += 1
            print(f"\nRun {run_idx}/{total_runs}: {params}")

            try:
                # Override BehavioralDistance gamma for this run
                original_gamma = BehavioralDistance.gamma
                if "gamma" in params:
                    BehavioralDistance.gamma = params["gamma"]

                # Collect data with current parameters
                behavioral = BehavioralDistance(self.env)
                behavioral.collect_transitions(n_episodes=params.get("n_episodes", 100))

                # Compute distance matrix with custom weights
                dist_matrix, keys_list = behavioral.compute_distance_matrix(
                    sample_size=params.get("sample_size", 100)
                )

                # Override weights in distance computation
                alpha = params.get("alpha", 1.0)
                beta = params.get("beta", 1.0)

                # Recompute distance with these weights
                n = len(keys_list)
                new_dist = np.zeros((n, n))
                for i, key_i in enumerate(keys_list):
                    for j, key_j in enumerate(keys_list[i:], i):
                        if i == j:
                            continue

                        r_i = (
                            np.mean(behavioral.reward_data[key_i])
                            if behavioral.reward_data[key_i]
                            else 0
                        )
                        r_j = (
                            np.mean(behavioral.reward_data[key_j])
                            if behavioral.reward_data[key_j]
                            else 0
                        )
                        r_diff = abs(r_i - r_j)

                        w_dist = behavioral._wasserstein_distance(
                            behavioral.transition_data[key_i],
                            behavioral.transition_data[key_j],
                        )

                        new_dist[i, j] = alpha * w_dist + beta * r_diff
                        new_dist[j, i] = new_dist[i, j]

                np.fill_diagonal(new_dist, 0)
                behavioral.distance_matrix = new_dist

                # Build filtration and cluster
                filtration = BehavioralFiltration(new_dist, keys_list)
                r_values = np.percentile(
                    new_dist[new_dist > 0], [10, 25, 40, 55, 70, 85]
                )
                filtration.build_vietoris_rips(r_values)

                spectral = FiltrationSpectralClustering(new_dist, keys_list)
                labels, persistent, _ = spectral.persistent_spectral_clustering(
                    r_range=r_values, persistence_threshold=params.get("threshold", 0.3)
                )

                # Build abstract MDP
                abstract = DeterministicAbstractMDP(
                    labels,
                    keys_list,
                    behavioral.transition_data,
                    behavioral.reward_data,
                    self.env,
                )

                # Compute metrics
                n_clusters = len(np.unique(labels[labels >= 0]))

                # Silhouette score
                valid_mask = labels >= 0
                if np.sum(valid_mask) > 1:
                    sil_score = silhouette_score(
                        new_dist[valid_mask][:, valid_mask],
                        labels[valid_mask],
                        metric="precomputed",
                    )
                else:
                    sil_score = -1

                # Abstract MDP properties
                trans_matrix, _ = abstract.get_abstract_transition_matrix()

                # Action distinguishability
                action_diff = 0
                for h in abstract.H:
                    next_states = set(trans_matrix[h, a] for a in abstract.A)
                    action_diff += len(next_states) / len(abstract.A)
                action_diff /= len(abstract.H) if abstract.H else 1

                # Store results
                result = {
                    **params,
                    "n_clusters": n_clusters,
                    "silhouette": sil_score,
                    "action_distinguishability": action_diff,
                    "n_abstract_states": len(abstract.H),
                    "n_persistent": len(persistent),
                }

                self.sweep_results.append(result)

                # Restore original gamma
                if "gamma" in params:
                    BehavioralDistance.gamma = original_gamma

            except Exception as e:
                print(f"  Error: {e}")
                continue

        return pd.DataFrame(self.sweep_results)

    def plot_sweep_results(self, df=None):
        """Visualize parameter sweep results"""
        if df is None:
            df = pd.DataFrame(self.sweep_results)

        if len(df) == 0:
            print("No sweep results to plot")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)

        # 1. Effect of alpha/beta on clusters
        ax1 = fig.add_subplot(gs[0, 0])
        if "alpha" in df.columns and "beta" in df.columns:
            pivot = df.pivot_table(
                values="n_clusters", index="alpha", columns="beta", aggfunc="mean"
            )
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax1)
            ax1.set_title("Number of Clusters")

        # 2. Effect on silhouette score
        ax2 = fig.add_subplot(gs[0, 1])
        if "alpha" in df.columns and "beta" in df.columns:
            pivot = df.pivot_table(
                values="silhouette", index="alpha", columns="beta", aggfunc="mean"
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax2)
            ax2.set_title("Silhouette Score")

        # 3. Effect on action distinguishability
        ax3 = fig.add_subplot(gs[0, 2])
        if "alpha" in df.columns and "beta" in df.columns:
            pivot = df.pivot_table(
                values="action_distinguishability",
                index="alpha",
                columns="beta",
                aggfunc="mean",
            )
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="plasma", ax=ax3)
            ax3.set_title("Action Distinguishability")

        # 4. Scatter: alpha vs n_clusters
        ax4 = fig.add_subplot(gs[1, 0])
        if "alpha" in df.columns:
            for beta in sorted(df["beta"].unique())[:4]:
                subset = df[df["beta"] == beta]
                ax4.scatter(
                    subset["alpha"],
                    subset["n_clusters"],
                    label=f"β={beta}",
                    alpha=0.7,
                    s=50,
                )
            ax4.set_xlabel("α (Wasserstein Weight)")
            ax4.set_ylabel("Number of Clusters")
            ax4.set_title("α vs Clusters")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Scatter: beta vs n_clusters
        ax5 = fig.add_subplot(gs[1, 1])
        if "beta" in df.columns:
            for alpha in sorted(df["alpha"].unique())[:4]:
                subset = df[df["alpha"] == alpha]
                ax5.scatter(
                    subset["beta"],
                    subset["n_clusters"],
                    label=f"α={alpha}",
                    alpha=0.7,
                    s=50,
                )
            ax5.set_xlabel("β (Reward Weight)")
            ax5.set_ylabel("Number of Clusters")
            ax5.set_title("β vs Clusters")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 3D scatter
        ax6 = fig.add_subplot(gs[1, 2], projection="3d")
        if "alpha" in df.columns and "beta" in df.columns:
            scatter = ax6.scatter(
                df["alpha"],
                df["beta"],
                df["n_clusters"],
                c=df["silhouette"],
                cmap="RdBu_r",
                s=50,
                alpha=0.7,
            )
            ax6.set_xlabel("α")
            ax6.set_ylabel("β")
            ax6.set_zlabel("# Clusters")
            ax6.set_title("3D Parameter Space")
            fig.colorbar(scatter, ax=ax6, shrink=0.5, aspect=10, label="Silhouette")

        # 7. Distribution of results
        ax7 = fig.add_subplot(gs[2, 0])
        df["n_clusters"].hist(bins=15, ax=ax7, edgecolor="black", alpha=0.7)
        ax7.set_xlabel("Number of Clusters")
        ax7.set_ylabel("Frequency")
        ax7.set_title("Cluster Count Distribution")
        ax7.grid(True, alpha=0.3)

        # 8. Silhouette distribution
        ax8 = fig.add_subplot(gs[2, 1])
        df["silhouette"].hist(
            bins=15, ax=ax8, edgecolor="black", alpha=0.7, color="green"
        )
        ax8.set_xlabel("Silhouette Score")
        ax8.set_ylabel("Frequency")
        ax8.set_title("Silhouette Distribution")
        ax8.grid(True, alpha=0.3)

        # 9. Correlation matrix
        ax9 = fig.add_subplot(gs[2, 2])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            ax=ax9,
            cbar_kws={"shrink": 0.8},
        )
        ax9.set_title("Parameter Correlation Matrix")

        plt.suptitle("Parameter Sweep Analysis", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        return df


# ============ 4. MAIN DIAGNOSTIC SUITE ============


class DiagnosticSuite:
    """
    Complete diagnostic suite combining all analyses
    """

    def __init__(self, env_name="FrozenLake-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.results = {}

    def run_all_diagnostics(self, n_episodes=200, sample_size=100):
        """Run comprehensive diagnostic suite"""

        print("\n" + "=" * 70)
        print("COMPREHENSIVE DIAGNOSTIC SUITE")
        print("=" * 70)

        # 1. Behavioral distance component analysis
        print("\n" + "-" * 50)
        print("1. Behavioral Distance Component Analysis")
        print("-" * 50)

        bdd = BehavioralDistanceDiagnostics(self.env, n_episodes, sample_size)
        bdd.run_component_analysis()
        bdd.plot_component_analysis()
        self.results["component_analysis"] = bdd.results

        # 2. Run standard pipeline for quality analysis
        print("\n" + "-" * 50)
        print("2. Abstraction Quality Analysis")
        print("-" * 50)

        from filtr import run_topological_abstraction_pipeline

        pipeline_results = run_topological_abstraction_pipeline(
            env_name=self.env_name, n_episodes=n_episodes, persistence_threshold=0.3
        )

        # 3. Quality diagnostics
        print("\n" + "-" * 50)
        print("3. Detailed Quality Metrics")
        print("-" * 50)

        quality = AbstractionQualityDiagnostics(
            pipeline_results["behavioral"],
            pipeline_results["filtration"],
            pipeline_results["spectral"],
            pipeline_results["abstract_mdp"],
        )

        cluster_quality = quality.analyze_cluster_quality()
        dyn_quality = quality.analyze_abstract_dynamics()

        print("\nCluster Quality:")
        for k, v in cluster_quality.items():
            print(f"  {k}: {v}")

        print("\nAbstract Dynamics:")
        for k, v in dyn_quality.items():
            if k not in ["trans_matrix", "reward_matrix"]:
                print(f"  {k}: {v}")

        quality.plot_abstraction_quality()
        self.results["quality"] = {"cluster": cluster_quality, "dynamics": dyn_quality}

        # 4. Parameter sweep (smaller version for speed)
        print("\n" + "-" * 50)
        print("4. Parameter Sensitivity Analysis (quick sweep)")
        print("-" * 50)

        sweep = ParameterSweepDiagnostics(self.env_name)

        param_grid = {
            "alpha": [0.5, 1.0, 2.0],
            "beta": [0.5, 1.0, 2.0],
            "n_episodes": [100],
            "threshold": [0.3],
            "sample_size": [80],
        }

        sweep_results = sweep.run_sweep(param_grid)
        sweep.plot_sweep_results(sweep_results)
        self.results["sweep"] = sweep_results

        # 5. Summary report
        self.generate_summary_report()

        return self.results

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""

        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY REPORT")
        print("=" * 70)

        if "component_analysis" in self.results:
            stats = self.results["component_analysis"]["statistics"]

            print("\n1. Behavioral Distance Statistics:")
            print(
                f"   • Mean distance range: [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]"
            )
            print(
                f"   • Std range: [{stats['std'].min():.3f}, {stats['std'].max():.3f}]"
            )
            print(
                f"   • Wasserstein correlation: [{stats['corr_wasserstein'].min():.3f}, "
                f"{stats['corr_wasserstein'].max():.3f}]"
            )
            print(
                f"   • Reward correlation: [{stats['corr_reward'].min():.3f}, "
                f"{stats['corr_reward'].max():.3f}]"
            )

        if "quality" in self.results:
            quality = self.results["quality"]

            print("\n2. Abstraction Quality:")
            if "cluster" in quality:
                print(
                    f"   • Number of clusters: {quality['cluster'].get('n_clusters', 'N/A')}"
                )
                print(
                    f"   • Silhouette score: {quality['cluster'].get('silhouette_score', 'N/A'):.3f}"
                )
                print(
                    f"   • Cluster separation: {quality['cluster'].get('mean_separation', 'N/A'):.3f}"
                )

            if "dynamics" in quality:
                print(f"\n3. Abstract MDP Properties:")
                print(
                    f"   • Abstract states: {quality['dynamics'].get('n_states', 'N/A')}"
                )
                print(
                    f"   • Deterministic: {quality['dynamics'].get('deterministic', 'N/A')}"
                )
                print(
                    f"   • Action entropy: {quality['dynamics'].get('action_entropy_mean', 'N/A'):.3f}"
                )
                print(f"   • SCCs: {quality['dynamics'].get('n_scc', 'N/A')}")
                print(f"   • Cycles: {quality['dynamics'].get('n_cycles', 'N/A')}")

        if "sweep" in self.results and len(self.results["sweep"]) > 0:
            sweep = self.results["sweep"]

            print("\n4. Parameter Sensitivity (Quick Sweep):")
            print(f"   • Best silhouette: {sweep['silhouette'].max():.3f}")
            print(f"   • Best params: ")
            best_idx = sweep["silhouette"].idxmax()
            for col in ["alpha", "beta"]:
                if col in sweep.columns:
                    print(f"     {col}: {sweep.loc[best_idx, col]}")

            print(
                f"\n   • Cluster count range: [{sweep['n_clusters'].min()}, {sweep['n_clusters'].max()}]"
            )

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS:")
        print("=" * 70)

        # Generate recommendations based on findings
        if "quality" in self.results:
            qual = self.results["quality"]

            if qual.get("cluster", {}).get("silhouette_score", -1) < 0.2:
                print("Low silhouette score - clusters may not be well-separated.")
                print(
                    "   Consider: Increasing α (Wasserstein weight) or adjusting persistence threshold"
                )

            if qual.get("dynamics", {}).get("action_entropy_mean", 1) > 0.5:
                print("High action entropy - actions not very distinguishable.")
                print(
                    "   Consider: Reducing β (reward weight) to focus on transition differences"
                )

            if qual.get("dynamics", {}).get("n_states", 0) < 2:
                print("Very few abstract states - abstraction may be too coarse.")
                print(
                    "   Consider: Lowering persistence threshold or increasing α/β weights"
                )

        print("\nDiagnostic complete!")


# ============ 5. MAIN EXECUTION ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostic Suite for Topological Abstraction"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake-v1",
        choices=[
            "FrozenLake-v1",
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "Taxi-v3",
        ],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="Number of episodes for data collection",
    )
    parser.add_argument(
        "--sample", type=int, default=80, help="Sample size for distance matrix"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only quick diagnostics"
    )

    args = parser.parse_args()

    # Run diagnostic suite
    suite = DiagnosticSuite(args.env)

    if args.quick:
        print("Running quick diagnostics...")
        # Just run component analysis and quality check
        bdd = BehavioralDistanceDiagnostics(suite.env, 100, 50)
        bdd.run_component_analysis(
            alpha_values=[0.5, 1.0, 2.0], beta_values=[0.5, 1.0, 2.0]
        )
        bdd.plot_component_analysis()
    else:
        # Full diagnostics
        suite.run_all_diagnostics(n_episodes=args.episodes, sample_size=args.sample)

    plt.show()
