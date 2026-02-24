import numpy as np
import gymnasium as gym
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json


class FrozenLakeQLearning:
    def __init__(
        self,
        env_name: str = "FrozenLake-v1",
        map_name: str = "4x4",
        is_slippery: bool = True,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        episodes: int = 20000,
    ):
        """
        Initialize Q-learning agent for FrozenLake
        """
        # Create environment
        self.env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery)
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery

        # Q-learning parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes

        # Initialize Q-table
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Tracking metrics
        self.rewards_history = []
        self.steps_history = []
        self.success_rate_history = []
        self.epsilon_history = []

    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy policy
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, verbose: bool = True) -> Dict:
        """
        Train the Q-learning agent
        """
        print(f"Training Q-learning on {self.map_name} FrozenLake...")
        print(f"Environment: slippery={self.is_slippery}")
        print(f"States: {self.n_states}, Actions: {self.n_actions}")
        print("-" * 50)

        success_window = []

        for episode in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # Choose action
                action = self.choose_action(state)

                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Q-learning update
                best_next_action = np.max(self.q_table[next_state])
                self.q_table[state, action] += self.lr * (
                    reward + self.gamma * best_next_action - self.q_table[state, action]
                )

                state = next_state
                total_reward += reward
                steps += 1

            # Track metrics
            self.rewards_history.append(total_reward)
            self.steps_history.append(steps)

            # Track success (reached goal)
            success = 1 if total_reward > 0 else 0
            success_window.append(success)
            if len(success_window) > 100:
                success_window.pop(0)

            success_rate = np.mean(success_window)
            self.success_rate_history.append(success_rate)
            self.epsilon_history.append(self.epsilon)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Print progress
            if verbose and (episode + 1) % 1000 == 0:
                print(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"Success Rate: {success_rate:.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        print("-" * 50)
        print(f"Training completed! Final success rate: {success_rate:.3f}")

        return {
            "q_table": self.q_table,
            "rewards_history": self.rewards_history,
            "success_rate_history": self.success_rate_history,
            "final_success_rate": success_rate,
        }

    def collect_trajectories(
        self, n_trajectories: int = 100, epsilon_greedy: bool = False
    ) -> List[Dict]:
        """
        Collect trajectories using the learned policy

        Args:
            n_trajectories: Number of trajectories to collect
            epsilon_greedy: If True, use epsilon-greedy policy (more exploration)
                          If False, use greedy policy (pure exploitation)

        Returns:
            List of trajectories, each containing states, actions, rewards, etc.
        """
        trajectories = []

        for episode in range(n_trajectories):
            state, _ = self.env.reset()
            done = False
            trajectory = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
                "total_reward": 0,
                "steps": 0,
            }

            while not done:
                # Choose action based on policy
                if epsilon_greedy and np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition
                trajectory["states"].append(state)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["next_states"].append(next_state)
                trajectory["dones"].append(done)

                state = next_state
                trajectory["total_reward"] += reward
                trajectory["steps"] += 1

            trajectories.append(trajectory)

        return trajectories

    def extract_state_values(self) -> np.ndarray:
        """
        Extract state values V(s) = max_a Q(s,a)
        """
        return np.max(self.q_table, axis=1)

    def extract_state_action_values(self) -> np.ndarray:
        """
        Extract state-action values Q(s,a)
        """
        return self.q_table.copy()

    def save_model(self, filepath: str):
        """
        Save the trained model and parameters
        """
        model_data = {
            "q_table": self.q_table,
            "parameters": {
                "env_name": self.env_name,
                "map_name": self.map_name,
                "is_slippery": self.is_slippery,
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "episodes": self.episodes,
            },
            "metrics": {
                "rewards_history": self.rewards_history,
                "success_rate_history": self.success_rate_history,
                "final_success_rate": (
                    self.success_rate_history[-1] if self.success_rate_history else 0
                ),
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.q_table = model_data["q_table"]

        # Restore parameters if they exist
        if "parameters" in model_data:
            params = model_data["parameters"]
            self.env_name = params.get("env_name", self.env_name)
            self.map_name = params.get("map_name", self.map_name)
            self.is_slippery = params.get("is_slippery", self.is_slippery)
            self.lr = params.get("learning_rate", self.lr)
            self.gamma = params.get("discount_factor", self.gamma)
            self.epsilon = params.get("epsilon", self.epsilon)
            self.epsilon_min = params.get("epsilon_min", self.epsilon_min)
            self.epsilon_decay = params.get("epsilon_decay", self.epsilon_decay)

        print(f"Model loaded from {filepath}")

    def plot_learning_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Success rate
        axes[0, 0].plot(self.success_rate_history)
        axes[0, 0].set_title("Success Rate Over Time")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Success Rate (100-episode moving avg)")
        axes[0, 0].grid(True, alpha=0.3)

        # Epsilon decay
        axes[0, 1].plot(self.epsilon_history)
        axes[0, 1].set_title("Epsilon Decay")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Epsilon")
        axes[0, 1].grid(True, alpha=0.3)

        # Rewards
        axes[1, 0].plot(self.rewards_history)
        axes[1, 0].set_title("Episode Rewards")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Total Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # Steps per episode
        axes[1, 1].plot(self.steps_history)
        axes[1, 1].set_title("Steps per Episode")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Steps")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def visualize_policy(self):
        """
        Visualize the learned policy on the grid
        """
        # Action mapping
        action_symbols = ["←", "↓", "→", "↑"]  # Left, Down, Right, Up

        # Get grid dimensions
        if self.map_name == "4x4":
            grid_size = 4
        else:  # 8x8
            grid_size = 8

        policy_grid = np.full((grid_size, grid_size), " ", dtype="<U2")

        for state in range(self.n_states):
            row = state // grid_size
            col = state % grid_size
            best_action = np.argmax(self.q_table[state])
            policy_grid[row, col] = action_symbols[best_action]

        print("\nLearned Policy:")
        print("-" * (grid_size * 4 + 1))
        for row in range(grid_size):
            row_str = "|"
            for col in range(grid_size):
                row_str += f" {policy_grid[row, col]} |"
            print(row_str)
            print("-" * (grid_size * 4 + 1))


def run_experiment():
    """
    Run a complete experiment with model saving and trajectory collection
    """
    # Create directories for saving results
    os.makedirs("models", exist_ok=True)
    os.makedirs("trajectories", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Experiment 1: Train on 4x4 FrozenLake (slippery)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: 4x4 FrozenLake (Slippery)")
    print("=" * 60)

    agent1 = FrozenLakeQLearning(
        env_name="FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        episodes=11000,
    )

    # Train
    results1 = agent1.train(verbose=True)

    # Save model
    agent1.save_model("models/frozen_lake_4x4_slippery.pkl")

    # Plot learning curves
    agent1.plot_learning_curves(save_path="plots/learning_curves_4x4.png")

    # Visualize policy
    agent1.visualize_policy()

    # Collect trajectories
    trajectories1 = agent1.collect_trajectories(
        n_trajectories=100, epsilon_greedy=False
    )

    # Save trajectories
    with open("trajectories/trajectories_4x4.pkl", "wb") as f:
        pickle.dump(trajectories1, f)

    # Extract values
    state_values1 = agent1.extract_state_values()
    state_action_values1 = agent1.extract_state_action_values()

    print(f"\nState values (first 10 states):")
    for i, v in enumerate(state_values1[:10]):
        print(f"  State {i}: {v:.3f}")

    # Experiment 2: Train on 8x8 FrozenLake (slippery)
    # print("\n" + "=" * 60)
    # print("EXPERIMENT 2: 8x8 FrozenLake (Slippery)")
    # print("=" * 60)

    # agent2 = FrozenLakeQLearning(
    #    env_name="FrozenLake-v1",
    #    map_name="8x8",
    #    is_slippery=True,
    #    learning_rate=0.1,
    #    discount_factor=0.99,
    #    epsilon=1.0,
    #    epsilon_min=0.01,
    #    epsilon_decay=0.998,  # Slower decay for larger environment
    #    episodes=20000,
    # )

    ## Train
    # results2 = agent2.train(verbose=True)

    ## Save model
    # agent2.save_model("models/frozen_lake_8x8_slippery.pkl")

    ## Plot learning curves
    # agent2.plot_learning_curves(save_path="plots/learning_curves_8x8.png")

    ## Visualize policy
    # agent2.visualize_policy()

    ## Collect trajectories
    # trajectories2 = agent2.collect_trajectories(
    #    n_trajectories=100, epsilon_greedy=False
    # )

    ## Save trajectories
    # with open("trajectories/trajectories_8x8.pkl", "wb") as f:
    #    pickle.dump(trajectories2, f)

    ## Save experiment summary
    summary = {
        "experiment1": {
            "final_success_rate": float(results1["final_success_rate"]),
            "parameters": {"map": "4x4", "is_slippery": True, "episodes": 10000},
        },
        # "experiment2": {
        #    "final_success_rate": float(results2["final_success_rate"]),
        #    "parameters": {"map": "8x8", "is_slippery": True, "episodes": 20000},
        # },
    }

    with open("trajectories/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"Models saved in: models/")
    print(f"Trajectories saved in: trajectories/")
    print(f"Plots saved in: plots/")

    return agent1, trajectories1


def load_and_replay(model_path: str, n_episodes: int = 10):
    """
    Load a saved model and replay episodes
    """
    # Create agent and load model
    agent = FrozenLakeQLearning()
    agent.load_model(model_path)

    print(f"\nReplaying {n_episodes} episodes with learned policy:")
    print("-" * 50)

    trajectories = agent.collect_trajectories(
        n_trajectories=n_episodes, epsilon_greedy=False
    )

    for i, traj in enumerate(trajectories):
        print(
            f"Episode {i + 1}: Steps={traj['steps']}, Total Reward={traj['total_reward']}"
        )

    return trajectories


# Example usage
if __name__ == "__main__":
    # Run full experiment
    agent1, traj1 = run_experiment()

    # Example: Load and replay a saved model
    print("\n" + "=" * 60)
    print("LOADING AND REPLAYING SAVED MODEL")
    print("=" * 60)
    load_and_replay("models/frozen_lake_4x4_slippery.pkl", n_episodes=5)

    # Example: Analyze trajectories
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS")
    print("=" * 60)

    # Calculate average trajectory length and reward
    avg_steps = np.mean([t["steps"] for t in traj1])
    avg_reward = np.mean([t["total_reward"] for t in traj1])
    success_rate = np.mean([1 if t["total_reward"] > 0 else 0 for t in traj1])

    print(f"4x4 FrozenLake - 100 trajectories:")
    print(f"  Average steps: {avg_steps:.2f}")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Success rate: {success_rate:.3f}")
