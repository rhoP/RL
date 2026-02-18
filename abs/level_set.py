import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.patches import Rectangle
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import warnings

warnings.filterwarnings("ignore")


# Create custom FrozenLake environment - access the unwrapped version
class FrozenLakeVisualizer:
    def __init__(self, map_name="4x4", is_slippery=True):
        # Create the environment
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

        # Get the unwrapped environment to access transition dynamics
        self.unwrapped = self.env.unwrapped

        self.n_states = self.unwrapped.observation_space.n
        self.n_actions = self.unwrapped.action_space.n
        self.grid_size = 4 if map_name == "4x4" else 8

        # Get the map description
        self.desc = self.unwrapped.desc

        # Transition dynamics are in the unwrapped environment
        self.P = self.unwrapped.P

    def state_to_pos(self, state):
        """Convert state index to grid position"""
        return state // self.grid_size, state % self.grid_size

    def pos_to_state(self, row, col):
        """Convert grid position to state index"""
        return row * self.grid_size + col


# Alternative: Direct FrozenLakeEnv without wrapper
class DirectFrozenLake:
    def __init__(self, map_name="4x4", is_slippery=True):
        # Use FrozenLakeEnv directly instead of gym.make
        self.env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.grid_size = 4 if map_name == "4x4" else 8
        self.desc = self.env.desc
        self.P = self.env.P  # Direct access to transition dynamics

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


# Initialize environment (using DirectFrozenLake to avoid TimeLimit wrapper)
env_viz = DirectFrozenLake(map_name="4x4", is_slippery=True)

# ============================================
# 1. DYNAMIC PROGRAMMING - VALUE ITERATION
# ============================================


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=100):
    """Value iteration algorithm with history tracking"""
    n_states = env.n_states
    n_actions = env.n_actions

    # Initialize value function
    V = np.zeros(n_states)
    V_history = [V.copy()]
    policy_history = []

    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            # Bellman update using transition dynamics
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, terminated in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            V_new[s] = np.max(action_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new.copy()
        V_history.append(V.copy())

        # Extract greedy policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, terminated in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )
            policy[s] = np.argmax(action_values)
        policy_history.append(policy.copy())

        if delta < theta:
            break

    return V_history, policy_history


# ============================================
# 2. Q-LEARNING (OFF-POLICY TD)
# ============================================


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500):
    """Q-learning algorithm with history tracking"""
    n_states = env.n_states
    n_actions = env.n_actions

    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))
    V_history = []  # Store value function (max over actions) every few steps
    Q_history = []

    value_snapshot_interval = 25
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            best_next_action = np.argmax(Q[next_state, :])
            td_target = reward + gamma * Q[next_state, best_next_action] * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        # Store value function snapshot
        if episode % value_snapshot_interval == 0:
            V = np.max(Q, axis=1)
            V_history.append(V.copy())
            Q_history.append(Q.copy())

    return V_history, Q_history, episode_rewards


# ============================================
# 3. SARSA (ON-POLICY TD)
# ============================================


def sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500):
    """SARSA algorithm with history tracking"""
    n_states = env.n_states
    n_actions = env.n_actions

    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))
    V_history = []
    Q_history = []

    value_snapshot_interval = 25
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()

        # Choose first action
        if np.random.random() < epsilon:
            action = env.env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Choose next action (for SARSA update)
            if np.random.random() < epsilon:
                next_action = env.env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            # SARSA update
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        # Store value function snapshot
        if episode % value_snapshot_interval == 0:
            V = np.max(Q, axis=1)
            V_history.append(V.copy())
            Q_history.append(Q.copy())

    return V_history, Q_history, episode_rewards


# ============================================
# 4. MONTE CARLO METHODS
# ============================================


def monte_carlo(env, gamma=0.99, epsilon=0.1, episodes=500):
    """First-visit Monte Carlo with history tracking"""
    n_states = env.n_states
    n_actions = env.n_actions

    # Initialize Q-table and returns
    Q = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    V_history = []
    episode_rewards = []

    value_snapshot_interval = 25

    for episode in range(episodes):
        # Generate an episode
        episode_memory = []
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            # Epsilon-soft policy
            if np.random.random() < epsilon:
                action = env.env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_memory.append((state, action, reward))
            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        # First-visit MC update
        G = 0
        visited_state_actions = set()

        for t in range(len(episode_memory) - 1, -1, -1):
            state, action, reward = episode_memory[t]
            G = gamma * G + reward

            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                returns_sum[state, action] += G
                returns_count[state, action] += 1
                Q[state, action] = (
                    returns_sum[state, action] / returns_count[state, action]
                )

        # Store value function snapshot
        if episode % value_snapshot_interval == 0:
            V = np.max(Q, axis=1)
            V_history.append(V.copy())

    return V_history, episode_rewards


# ============================================
# 5. DYNAMIC PROGRAMMING - POLICY ITERATION
# ============================================


def policy_iteration(env, gamma=0.99, max_iterations=50):
    """Policy iteration algorithm with history tracking"""
    n_states = env.n_states
    n_actions = env.n_actions

    # Initialize random policy
    policy = np.random.randint(0, n_actions, n_states)
    V_history = [np.zeros(n_states)]
    policy_history = [policy.copy()]

    for iteration in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            V = V_history[-1].copy()

            for s in range(n_states):
                v = V[s]
                # Use current policy
                a = policy[s]
                new_v = 0
                for prob, next_state, reward, terminated in env.P[s][a]:
                    new_v += prob * (reward + gamma * V[next_state] * (not terminated))
                V[s] = new_v
                delta = max(delta, abs(v - V[s]))

            if delta < 1e-6:
                break

        V_history.append(V.copy())

        # Policy Improvement
        policy_stable = True
        for s in range(n_states):
            old_action = policy[s]

            # Compute action values
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, terminated in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            policy[s] = np.argmax(action_values)

            if old_action != policy[s]:
                policy_stable = False

        policy_history.append(policy.copy())

        if policy_stable:
            break

    return V_history, policy_history


# Run all algorithms
print("Running algorithms...")

# Dynamic Programming - Value Iteration
vi_history, vi_policy = value_iteration(env_viz, max_iterations=50)

# Q-Learning
ql_history, ql_q, ql_rewards = q_learning(env_viz, episodes=300)

# SARSA
sarsa_history, sarsa_q, sarsa_rewards = sarsa(env_viz, episodes=300)

# Monte Carlo
mc_history, mc_rewards = monte_carlo(env_viz, episodes=300)

# Policy Iteration
pi_history, pi_policy = policy_iteration(env_viz)

# ============================================
# VISUALIZATION
# ============================================

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# 1. Convergence comparison
ax1 = plt.subplot(3, 3, 1)
iterations = min(len(vi_history), 50)
ax1.plot(
    range(iterations),
    [np.mean(v) for v in vi_history[:iterations]],
    "b-",
    linewidth=2,
    label="Value Iteration",
)
ax1.plot(
    range(len(ql_history)),
    [np.mean(v) for v in ql_history],
    "r-",
    linewidth=2,
    label="Q-Learning",
)
ax1.plot(
    range(len(sarsa_history)),
    [np.mean(v) for v in sarsa_history],
    "g-",
    linewidth=2,
    label="SARSA",
)
ax1.plot(
    range(len(mc_history)),
    [np.mean(v) for v in mc_history],
    "orange",
    linewidth=2,
    label="Monte Carlo",
)
ax1.plot(
    range(min(len(pi_history), 20)),
    [np.mean(v) for v in pi_history[:20]],
    "purple",
    linewidth=2,
    label="Policy Iteration",
)
ax1.set_xlabel("Iteration / Episode (x25 for TD methods)")
ax1.set_ylabel("Average Value")
ax1.set_title("Convergence of Value Functions")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Learning curves (rewards)
ax2 = plt.subplot(3, 3, 2)
window = 20
ql_smoothed = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
sarsa_smoothed = np.convolve(sarsa_rewards, np.ones(window) / window, mode="valid")
mc_smoothed = np.convolve(mc_rewards, np.ones(window) / window, mode="valid")

ax2.plot(ql_smoothed, "r-", linewidth=2, label="Q-Learning")
ax2.plot(sarsa_smoothed, "g-", linewidth=2, label="SARSA")
ax2.plot(mc_smoothed, "orange", linewidth=2, label="Monte Carlo")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Average Reward")
ax2.set_title("Learning Curves (Moving Average)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3-7: Final value functions as heatmaps
algorithms = [
    ("Value Iteration", vi_history[-1]),
    ("Q-Learning", ql_history[-1] if ql_history else np.zeros(16)),
    ("SARSA", sarsa_history[-1] if sarsa_history else np.zeros(16)),
    ("Monte Carlo", mc_history[-1] if mc_history else np.zeros(16)),
    ("Policy Iteration", pi_history[-1]),
]

for idx, (name, values) in enumerate(algorithms):
    ax = plt.subplot(3, 3, idx + 3)

    # Reshape to grid
    value_grid = values.reshape(4, 4)

    # Create heatmap
    im = ax.imshow(value_grid, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")

    # Add grid lines
    for i in range(5):
        ax.axhline(i - 0.5, color="white", linewidth=1)
        ax.axvline(i - 0.5, color="white", linewidth=1)

    # Add state labels and values
    for i in range(4):
        for j in range(4):
            cell_type = env_viz.desc[i][j].decode()

            # Background color for special states
            if cell_type == "H":
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="red", alpha=0.3)
                )
                ax.text(
                    j,
                    i,
                    "H",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )
            elif cell_type == "G":
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="gold", alpha=0.3)
                )
                ax.text(
                    j,
                    i,
                    "G",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )
            elif cell_type == "S":
                ax.text(
                    j,
                    i - 0.25,
                    "S",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

            # Add value text with appropriate color based on background
            text_color = (
                "white" if value_grid[i, j] < 0.6 or cell_type == "H" else "black"
            )
            y_offset = 0.1 if cell_type == "S" else 0
            ax.text(
                j,
                i + y_offset,
                f"{value_grid[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    ax.set_title(f"{name} - Final Values")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(
    "Comparison of RL Algorithms on FrozenLake", fontsize=16, fontweight="bold"
)
plt.tight_layout()
plt.show()

# ============================================
# ANIMATION OF VALUE FUNCTION EVOLUTION
# ============================================

# Create animation for each algorithm
fig_anim, axes_anim = plt.subplots(2, 3, figsize=(15, 10))
axes_anim = axes_anim.flatten()

# Prepare data for animation
anim_data = [
    ("Value Iteration", vi_history, "viridis"),
    ("Q-Learning", ql_history, "plasma"),
    ("SARSA", sarsa_history, "magma"),
    ("Monte Carlo", mc_history, "inferno"),
    ("Policy Iteration", pi_history, "cividis"),
]


def update_frame(frame):
    for idx, (name, history, cmap) in enumerate(anim_data):
        ax = axes_anim[idx]
        ax.clear()

        # Get current values (with bounds checking)
        if frame < len(history):
            values = history[frame]
        else:
            values = history[-1]

        # Reshape to grid
        value_grid = values.reshape(4, 4)

        # Create heatmap
        im = ax.imshow(value_grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

        # Add grid lines
        for i in range(5):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
            ax.axvline(i - 0.5, color="white", linewidth=0.5)

        # Add state labels
        for i in range(4):
            for j in range(4):
                cell_type = env_viz.desc[i][j].decode()

                # Background for special states
                if cell_type == "H":
                    ax.add_patch(
                        Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="red", alpha=0.3)
                    )
                elif cell_type == "G":
                    ax.add_patch(
                        Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="gold", alpha=0.3)
                    )
                elif cell_type == "S":
                    ax.text(
                        j,
                        i - 0.25,
                        "S",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )

                # Add value text
                if not (cell_type == "H" or cell_type == "G"):
                    text_color = "white" if value_grid[i, j] < 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value_grid[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                    )

        # Add iteration/episode info
        iter_type = (
            "Iteration"
            if name in ["Value Iteration", "Policy Iteration"]
            else "Episode (x25)"
        )
        ax.set_title(f"{name}\n{iter_type}: {frame}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar to the last subplot
    plt.colorbar(im, ax=axes_anim[-1], fraction=0.046, pad=0.04)

    return axes_anim


# Create animation
max_frames = max(len(h) for _, h, _ in anim_data)
anim = FuncAnimation(
    fig_anim, update_frame, frames=max_frames, interval=300, repeat=True
)

plt.tight_layout()
plt.show()

# ============================================
# POLICY VISUALIZATION
# ============================================

# Extract policies
_, vi_policy = value_iteration(env_viz, max_iterations=50)
_, pi_policy = policy_iteration(env_viz)

# Get final policies
ql_final_policy = (
    np.argmax(ql_q[-1], axis=1) if len(ql_q) > 0 else np.zeros(env_viz.n_states)
)
sarsa_final_policy = (
    np.argmax(sarsa_q[-1], axis=1) if len(sarsa_q) > 0 else np.zeros(env_viz.n_states)
)
# For MC, we'll use QL as proxy since MC doesn't store Q history in this implementation
mc_final_policy = ql_final_policy.copy()

# Direction mapping
action_arrows = ["←", "↓", "→", "↑"]  # LEFT, DOWN, RIGHT, UP
action_names = ["LEFT", "DOWN", "RIGHT", "UP"]

fig_policies, axes_policies = plt.subplots(2, 3, figsize=(15, 10))
axes_policies = axes_policies.flatten()

policies = [
    ("Value Iteration", vi_policy[-1] if len(vi_policy) > 0 else np.zeros(16)),
    ("Policy Iteration", pi_policy[-1] if len(pi_policy) > 0 else np.zeros(16)),
    ("Q-Learning", ql_final_policy),
    ("SARSA", sarsa_final_policy),
    ("Monte Carlo", mc_final_policy),
]

for idx, (name, policy) in enumerate(policies):
    ax = axes_policies[idx]

    # Create grid background
    grid = np.zeros((4, 4))
    im = ax.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    # Add policy arrows and cell information
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            cell_type = env_viz.desc[i][j].decode()

            # Background for special states
            if cell_type == "H":
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="red", alpha=0.5)
                )
                ax.text(
                    j,
                    i,
                    "HOLE",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=8,
                )
            elif cell_type == "G":
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="gold", alpha=0.5)
                )
                ax.text(
                    j,
                    i,
                    "GOAL",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=8,
                )
            elif cell_type == "S":
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="lightblue", alpha=0.3
                    )
                )
                ax.text(
                    j - 0.25,
                    i + 0.25,
                    "S",
                    ha="center",
                    va="center",
                    color="blue",
                    fontweight="bold",
                )
                # Add arrow for start state
                if state < len(policy):
                    action = int(policy[state])
                    ax.text(
                        j,
                        i,
                        action_arrows[action],
                        ha="center",
                        va="center",
                        color="blue",
                        fontsize=16,
                    )
            else:
                # Add policy arrow for regular states
                if state < len(policy):
                    action = int(policy[state])
                    ax.text(
                        j,
                        i,
                        action_arrows[action],
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=20,
                    )

    ax.set_title(f"{name} Policy")
    ax.set_xticks([])
    ax.set_yticks([])

# Add legend
ax_legend = axes_policies[-1]
ax_legend.axis("off")
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.5, label="Hole"),
    plt.Rectangle((0, 0), 1, 1, facecolor="gold", alpha=0.5, label="Goal"),
    plt.Rectangle((0, 0), 1, 1, facecolor="lightblue", alpha=0.3, label="Start"),
]
for i, (arrow, name) in enumerate(zip(action_arrows, action_names)):
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=arrow,
            color="w",
            markerfacecolor="black",
            markersize=15,
            label=name,
        )
    )
ax_legend.legend(handles=legend_elements, loc="center", fontsize=10)

plt.suptitle("Learned Policies Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("ALGORITHM COMPARISON SUMMARY")
print("=" * 60)

print("\nFinal Average Values:")
print(f"Value Iteration : {np.mean(vi_history[-1]):.4f}")
print(f"Policy Iteration: {np.mean(pi_history[-1]):.4f}")
print(f"Q-Learning      : {np.mean(ql_history[-1]) if ql_history else 0:.4f}")
print(f"SARSA           : {np.mean(sarsa_history[-1]) if sarsa_history else 0:.4f}")
print(f"Monte Carlo     : {np.mean(mc_history[-1]) if mc_history else 0:.4f}")

print("\nValue at Start State (0,0):")
print(f"Value Iteration : {vi_history[-1][0]:.4f}")
print(f"Policy Iteration: {pi_history[-1][0]:.4f}")
print(f"Q-Learning      : {ql_history[-1][0] if ql_history else 0:.4f}")
print(f"SARSA           : {sarsa_history[-1][0] if sarsa_history else 0:.4f}")
print(f"Monte Carlo     : {mc_history[-1][0] if mc_history else 0:.4f}")

print("\nOptimal Policy at Start State (action):")
action_names = ["LEFT", "DOWN", "RIGHT", "UP"]
print(f"Value Iteration : {action_names[int(vi_policy[-1][0])]}")
print(f"Policy Iteration: {action_names[int(pi_policy[-1][0])]}")
print(f"Q-Learning      : {action_names[int(ql_final_policy[0])]}")
print(f"SARSA           : {action_names[int(sarsa_final_policy[0])]}")
print(f"Monte Carlo     : {action_names[int(mc_final_policy[0])]}")

# Additional analysis: Value propagation visualization
fig_propagation, axes_prop = plt.subplots(1, 3, figsize=(15, 5))

# Show value function at different stages for Value Iteration
stages = [0, len(vi_history) // 3, len(vi_history) - 1]
stage_names = ["Initial", "Mid", "Final"]

for idx, (stage, name) in enumerate(zip(stages, stage_names)):
    ax = axes_prop[idx]
    values = vi_history[stage].reshape(4, 4)

    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")

    # Add grid and labels
    for i in range(4):
        for j in range(4):
            cell_type = env_viz.desc[i][j].decode()
            if cell_type == "H":
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="red", alpha=0.3)
                )
            elif cell_type == "G":
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="gold", alpha=0.3)
                )

            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if values[i, j] < 0.6 else "black",
                fontsize=9,
            )

    ax.set_title(f"Value Iteration - {name} Stage")
    ax.set_xticks([])
    ax.set_yticks([])

plt.colorbar(im, ax=axes_prop[-1])
plt.suptitle("Value Propagation Through Iterations", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
