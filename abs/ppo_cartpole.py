import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# ==================== STANDARD PPO IMPLEMENTATION ====================


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.features(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


class PPOAgent:
    """Standard PPO agent without hierarchical structure"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.action_dim = action_dim

        # Initialize network
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_action(self, state: np.ndarray) -> tuple:
        """Select action and return log probability and value"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store trajectory data"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value: float):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        values = self.values + [last_value]

        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + self.gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def update(self, last_value: float):
        """Update policy using PPO objective"""
        # Compute advantages and returns
        advantages, returns = self.compute_gae(last_value)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        advantages = torch.FloatTensor(np.array(advantages))
        returns = torch.FloatTensor(np.array(returns))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        for _ in range(self.epochs):
            # Generate random mini-batches
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get current policy probabilities and values
                action_probs, state_values = self.network(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratio and surrogate loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )

                # Compute losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                entropy_loss = -0.01 * entropy  # Entropy bonus for exploration

                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

        # Clear trajectory buffer
        self.clear_buffer()

    def clear_buffer(self):
        """Clear trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


# ==================== HIERARCHICAL AGENT (Simplified for Comparison) ====================


class HierarchicalNetwork(nn.Module):
    """Simplified hierarchical network for fair comparison with PPO"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Low-level policy network
        self.low_level = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # High-level value network (abstract state value)
        self.high_level = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Priority modulation network
        self.priority_mod = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for high-level value
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Output modulation factor between -1 and 1
        )

    def forward(self, state, high_value=None):
        # Get base action probabilities from low level
        base_probs = self.low_level(state)

        if high_value is not None:
            # Get priority modulation
            combined = torch.cat([state, high_value], dim=-1)
            modulation = self.priority_mod(combined)

            # Apply modulation (bias actions toward higher value abstract states)
            # This is a simplified version of the topological guidance
            modulated_probs = base_probs + 0.1 * modulation
            modulated_probs = torch.softmax(modulated_probs, dim=-1)
            return modulated_probs
        else:
            return base_probs


class SimpleHierarchicalAgent:
    """Simplified hierarchical agent for fair comparison"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_low: float = 3e-4,
        lr_high: float = 3e-4,
        gamma: float = 0.99,
    ):
        self.gamma = gamma
        self.action_dim = action_dim

        # Initialize networks
        self.network = HierarchicalNetwork(state_dim, action_dim)
        self.optimizer_low = optim.Adam(
            list(self.network.low_level.parameters())
            + list(self.network.priority_mod.parameters()),
            lr=lr_low,
        )
        self.optimizer_high = optim.Adam(
            self.network.high_level.parameters(), lr=lr_high
        )

        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.high_values = []

        # For tracking abstract state guidance
        self.abstract_state_buffer = deque(maxlen=100)

    def get_action(self, state: np.ndarray) -> int:
        """Get action with high-level guidance"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            # Get high-level value for this state
            high_value = self.network.high_level(state_tensor)

            # Get modulated action probabilities
            action_probs = self.network(state_tensor, high_value)
            dist = Categorical(action_probs)
            action = dist.sample()

        return action.item(), high_value.item()

    def store_transition(self, state, action, reward, high_value, done):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.high_values.append(high_value)
        self.dones.append(done)

        # Store for abstract state analysis (simplified)
        self.abstract_state_buffer.append((state, reward))

    def update(self):
        """Update both policies"""
        if len(self.states) < 32:
            return

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        high_values = torch.FloatTensor(np.array(self.high_values)).unsqueeze(1)
        dones = torch.FloatTensor(np.array(self.dones))

        # Compute returns for high-level value function
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # Update high-level value network
        high_loss = nn.MSELoss()(high_values, returns)
        self.optimizer_high.zero_grad()
        high_loss.backward()
        self.optimizer_high.step()

        # Update low-level policy with high-level guidance
        # Recompute high values with updated network
        with torch.no_grad():
            updated_high_values = self.network.high_level(states)

        # Get action probabilities with guidance
        action_probs = self.network(states, updated_high_values)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        # Simple policy gradient with baseline
        advantage = returns - updated_high_values

        # Policy loss
        policy_loss = -(log_probs * advantage.detach().squeeze()).mean()

        # Add entropy bonus for exploration
        entropy = dist.entropy().mean()
        total_loss = policy_loss - 0.01 * entropy

        self.optimizer_low.zero_grad()
        total_loss.backward()
        self.optimizer_low.step()

        # Clear buffer
        self.clear_buffer()

    def clear_buffer(self):
        """Clear trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.high_values = []
        self.dones = []


# ==================== TRAINING FUNCTIONS ====================


def train_ppo(env_name: str = "CartPole-v1", episodes: int = 500):
    """Train standard PPO agent"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action
            action, log_prob, value = agent.get_action(state)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)

            episode_reward += reward
            state = next_state

        # Get last value for bootstrapping
        with torch.no_grad():
            _, last_value = agent.network(torch.FloatTensor(next_state).unsqueeze(0))
            last_value = last_value.item()

        # Update agent
        agent.update(last_value)
        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"PPO - Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    env.close()
    return episode_rewards


def train_hierarchical(env_name: str = "CartPole-v1", episodes: int = 500):
    """Train hierarchical agent"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = SimpleHierarchicalAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action with high-level guidance
            action, high_value = agent.get_action(state)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, high_value, done)

            episode_reward += reward
            state = next_state

        # Update agent
        agent.update()
        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Hierarchical - Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    env.close()
    return episode_rewards


# ==================== COMPARISON AND VISUALIZATION ====================


def run_comparison(env_name: str = "CartPole-v1", episodes: int = 500, runs: int = 3):
    """
    Run multiple training runs for both agents and compare performance
    """
    print(f"Running comparison on {env_name} for {episodes} episodes...")
    print("=" * 50)

    all_ppo_rewards = []
    all_hier_rewards = []

    for run in range(runs):
        print(f"\nRun {run + 1}/{runs}")
        print("-" * 30)

        # Train PPO
        print("Training PPO...")
        # ppo_rewards = train_ppo(env_name, episodes)
        # all_ppo_rewards.append(ppo_rewards)

        # Train Hierarchical
        print("\nTraining Hierarchical Agent...")
        hier_rewards = train_hierarchical(env_name, episodes)
        all_hier_rewards.append(hier_rewards)

    # Convert to numpy arrays
    ppo_array = np.array(all_ppo_rewards)
    hier_array = np.array(all_hier_rewards)

    # Calculate statistics
    ppo_mean = np.mean(ppo_array, axis=0)
    ppo_std = np.std(ppo_array, axis=0)
    hier_mean = np.mean(hier_array, axis=0)
    hier_std = np.std(hier_array, axis=0)

    # Calculate moving averages for smoother curves
    window = 20
    ppo_smooth = np.convolve(ppo_mean, np.ones(window) / window, mode="valid")
    hier_smooth = np.convolve(hier_mean, np.ones(window) / window, mode="valid")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Raw training curves
    ax = axes[0, 0]
    ax.plot(ppo_mean, label="PPO", alpha=0.7, color="blue")
    ax.fill_between(
        range(len(ppo_mean)),
        ppo_mean - ppo_std,
        ppo_mean + ppo_std,
        alpha=0.2,
        color="blue",
    )
    ax.plot(hier_mean, label="Hierarchical", alpha=0.7, color="red")
    ax.fill_between(
        range(len(hier_mean)),
        hier_mean - hier_std,
        hier_mean + hier_std,
        alpha=0.2,
        color="red",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Curves (Raw)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Smoothed curves
    ax = axes[0, 1]
    x_smooth = range(window - 1, len(ppo_mean))
    ax.plot(x_smooth, ppo_smooth, label="PPO", color="blue", linewidth=2)
    ax.plot(x_smooth, hier_smooth, label="Hierarchical", color="red", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Training Curves (Moving Average, window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot of final performance
    ax = axes[1, 0]
    final_window = 100
    ppo_final = np.mean(ppo_array[:, -final_window:], axis=1)
    hier_final = np.mean(hier_array[:, -final_window:], axis=1)

    box_data = [ppo_final, hier_final]
    bp = ax.boxplot(box_data, labels=["PPO", "Hierarchical"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax.set_ylabel(f"Average Reward (last {final_window} episodes)")
    ax.set_title("Final Performance Distribution")
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning speed comparison
    ax = axes[1, 1]
    threshold = 200  # CartPole threshold

    ppo_speed = []
    hier_speed = []

    for run in range(runs):
        ppo_run = ppo_array[run]
        hier_run = hier_array[run]

        # Find first episode where moving average crosses threshold
        ppo_ma = np.convolve(ppo_run, np.ones(20) / 20, mode="valid")
        hier_ma = np.convolve(hier_run, np.ones(20) / 20, mode="valid")

        ppo_cross = np.where(ppo_ma >= threshold)[0]
        hier_cross = np.where(hier_ma >= threshold)[0]

        ppo_speed.append(ppo_cross[0] if len(ppo_cross) > 0 else episodes)
        hier_speed.append(hier_cross[0] if len(hier_cross) > 0 else episodes)

    x_pos = [1, 2]
    ax.bar(
        x_pos,
        [np.mean(ppo_speed), np.mean(hier_speed)],
        yerr=[np.std(ppo_speed), np.std(hier_speed)],
        capsize=10,
        color=["lightblue", "lightcoral"],
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["PPO", "Hierarchical"])
    ax.set_ylabel(f"Episodes to reach {threshold} reward")
    ax.set_title("Learning Speed Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("ppo_vs_hierarchical_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"\nPPO - Final Performance (last {final_window} episodes):")
    print(f"  Mean: {np.mean(ppo_final):.2f} +/- {np.std(ppo_final):.2f}")
    print(f"  Best Run: {np.max(ppo_final):.2f}")
    print(f"  Worst Run: {np.min(ppo_final):.2f}")

    print(f"\nHierarchical - Final Performance (last {final_window} episodes):")
    print(f"  Mean: {np.mean(hier_final):.2f} +/- {np.std(hier_final):.2f}")
    print(f"  Best Run: {np.max(hier_final):.2f}")
    print(f"  Worst Run: {np.min(hier_final):.2f}")

    print(f"\nLearning Speed (episodes to reach {threshold} reward):")
    print(f"  PPO: {np.mean(ppo_speed):.1f} +/- {np.std(ppo_speed):.1f}")
    print(f"  Hierarchical: {np.mean(hier_speed):.1f} +/- {np.std(hier_speed):.1f}")

    # Statistical test
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(ppo_final, hier_final)
    print(f"\nStatistical Test (t-test):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  Result: Significant difference between algorithms")
    else:
        print("  Result: No significant difference between algorithms")

    return {
        "ppo": ppo_array,
        "hierarchical": hier_array,
        "ppo_mean": ppo_mean,
        "hier_mean": hier_mean,
        "ppo_std": ppo_std,
        "hier_std": hier_std,
    }


def plot_detailed_comparison(results: dict, env_name: str = "CartPole-v1"):
    """
    Create a more detailed visualization of the comparison
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ppo_array = results["ppo"]
    hier_array = results["hierarchical"]

    # Plot 1: Individual runs (PPO)
    ax = axes[0, 0]
    for i in range(ppo_array.shape[0]):
        ax.plot(ppo_array[i], alpha=0.5, label=f"Run {i + 1}" if i == 0 else "")
    ax.plot(np.mean(ppo_array, axis=0), color="blue", linewidth=2, label="Mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("PPO - Individual Runs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Individual runs (Hierarchical)
    ax = axes[0, 1]
    for i in range(hier_array.shape[0]):
        ax.plot(hier_array[i], alpha=0.5, label=f"Run {i + 1}" if i == 0 else "")
    ax.plot(np.mean(hier_array, axis=0), color="red", linewidth=2, label="Mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Hierarchical - Individual Runs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative rewards comparison
    ax = axes[0, 2]
    ppo_cumsum = np.cumsum(np.mean(ppo_array, axis=0))
    hier_cumsum = np.cumsum(np.mean(hier_array, axis=0))
    ax.plot(ppo_cumsum, label="PPO", color="blue")
    ax.plot(hier_cumsum, label="Hierarchical", color="red")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Performance distribution over time
    ax = axes[1, 0]
    episodes = np.arange(ppo_array.shape[1])
    ppo_percentiles = np.percentile(ppo_array, [25, 50, 75], axis=0)
    hier_percentiles = np.percentile(hier_array, [25, 50, 75], axis=0)

    ax.fill_between(
        episodes,
        ppo_percentiles[0],
        ppo_percentiles[2],
        alpha=0.3,
        color="blue",
        label="PPO IQR",
    )
    ax.plot(episodes, ppo_percentiles[1], color="blue", linewidth=2, label="PPO Median")
    ax.fill_between(
        episodes,
        hier_percentiles[0],
        hier_percentiles[2],
        alpha=0.3,
        color="red",
        label="Hierarchical IQR",
    )
    ax.plot(
        episodes,
        hier_percentiles[1],
        color="red",
        linewidth=2,
        label="Hierarchical Median",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Performance Distribution Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Success rate over time
    ax = axes[1, 1]
    threshold = 200
    ppo_success = np.mean(ppo_array >= threshold, axis=0)
    hier_success = np.mean(hier_array >= threshold, axis=0)

    ax.plot(ppo_success, label="PPO", color="blue")
    ax.plot(hier_success, label="Hierarchical", color="red")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success Rate (Reward ≥ {threshold})")
    ax.set_title("Success Rate Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Improvement rate (derivative of smoothed curve)
    ax = axes[1, 2]
    window = 20
    ppo_smooth = np.convolve(
        np.mean(ppo_array, axis=0), np.ones(window) / window, mode="valid"
    )
    hier_smooth = np.convolve(
        np.mean(hier_array, axis=0), np.ones(window) / window, mode="valid"
    )

    ppo_improve = np.diff(ppo_smooth)
    hier_improve = np.diff(hier_smooth)

    x_improve = range(window, window + len(ppo_improve))
    ax.plot(x_improve, ppo_improve, label="PPO", color="blue", alpha=0.7)
    ax.plot(x_improve, hier_improve, label="Hierarchical", color="red", alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward Improvement Rate")
    ax.set_title("Learning Rate Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Detailed Comparison: PPO vs Hierarchical Agent on {env_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("ppo_vs_hierarchical_detailed.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run comparison
    print("=" * 60)
    print("PPO vs Hierarchical Agent Comparison")
    print("=" * 60)

    # Run multiple seeds for statistical significance
    results = run_comparison(
        env_name="CartPole-v1",
        episodes=500,
        runs=3,  # Number of independent runs
    )

    # Create detailed visualization
    plot_detailed_comparison(results, "CartPole-v1")

    print("\nComparison complete! Check the generated plots for visualization.")
