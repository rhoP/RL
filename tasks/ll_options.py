import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List, Optional


class Option:
    """Base class for options in the options framework"""

    def __init__(self, name: str, termination_condition):
        self.name = name
        self.termination_condition = termination_condition

    def policy(self, state: np.ndarray) -> np.ndarray:
        """Option policy - to be overridden by specific options"""
        raise NotImplementedError

    def is_terminated(self, state: np.ndarray) -> bool:
        """Check if option should terminate"""
        return self.termination_condition(state)


class FlipOption(Option):
    """Option 1: Flip the lander 180 degrees"""

    def __init__(self):
        super().__init__("Flip", self.flip_termination)
        self.target_angle = np.pi  # 180 degrees

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Policy to flip the lander 180 degrees
        State indices: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
        """
        angle = state[4]  # Current angle
        angular_velocity = state[5]

        action = np.array([0.0, 0.0, 0.0])  # [main, left, right]

        # Calculate angle error (wrap around to [-pi, pi])
        angle_error = self.target_angle - angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # PD controller for angle
        kp = 2.0
        kd = 0.5
        control = kp * angle_error - kd * angular_velocity

        # Apply control to side engines
        if control > 0:
            action[1] = min(abs(control), 1.0)  # Left engine
        else:
            action[2] = min(abs(control), 1.0)  # Right engine

        return action

    def flip_termination(self, state: np.ndarray) -> bool:
        """Terminate when we're close to 180 degrees"""
        angle = state[4]
        angle_error = abs((angle - self.target_angle + np.pi) % (2 * np.pi) - np.pi)
        return angle_error < 0.1  # Within ~5.7 degrees


class AccelerateDescentOption(Option):
    """Option 2: Accelerate descent using center engine in flipped state"""

    def __init__(self, target_y: float = 0.6):
        super().__init__("AccelerateDescent", self.accelerate_termination)
        self.target_y = target_y
        self.max_main_thrust = 1.0

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Policy to accelerate downward using center engine
        """
        y = state[1]  # Current height
        vy = state[3]  # Vertical velocity
        angle = state[4]

        action = np.array([0.0, 0.0, 0.0])

        # Only use main engine if we're roughly upside down
        if abs(angle - np.pi) < 0.3:  # Within ~17 degrees of upside down
            # Use main engine proportionally to distance from target
            thrust = min(1.0, max(0.1, (y - self.target_y) * 0.5))
            action[0] = thrust

        return action

    def accelerate_termination(self, state: np.ndarray) -> bool:
        """Terminate when we're near the target height"""
        y = state[1]
        return y < self.target_y


class DecelerateOption(Option):
    """Option 3: Decelerate for landing"""

    def __init__(self):
        super().__init__("Decelerate", self.decelerate_termination)

    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Policy to decelerate for soft landing
        """
        vy = state[3]  # Vertical velocity
        y = state[1]  # Height

        action = np.array([0.0, 0.0, 0.0])

        # PD controller for vertical velocity
        target_vy = -0.2  # Target descent rate for final approach
        vy_error = target_vy - vy

        # Increase thrust as we get lower
        height_factor = min(1.0, max(0.2, y * 2))
        thrust = min(1.0, max(0.0, vy_error * 0.5 * height_factor))

        action[0] = thrust

        # Add attitude control to stay upright-ish for landing
        angle = state[4]
        angle_target = 0.0  # Upright for landing
        angle_error = angle_target - angle

        # Small attitude corrections
        if abs(angle_error) > 0.1:
            if angle_error > 0:
                action[1] = min(abs(angle_error), 0.3)  # Left engine
            else:
                action[2] = min(abs(angle_error), 0.3)  # Right engine

        return action

    def decelerate_termination(self, state: np.ndarray) -> bool:
        """Terminate when landed"""
        return state[6] == 1 and state[7] == 1  # Both legs touching


class OptionPolicy(nn.Module):
    """High-level policy that selects options"""

    def __init__(self, state_dim: int, num_options: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_options),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def select_option(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy option selection"""
        if random.random() < epsilon:
            return random.randint(0, 2)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_values = self(state_tensor)
            return option_values.argmax().item()


class OptionCriticAgent:
    """Agent that learns using the options framework"""

    def __init__(self, state_dim: int):
        self.option_policy = OptionPolicy(state_dim, 3)
        self.options = [
            FlipOption(),
            AccelerateDescentOption(target_y=0.6),
            DecelerateOption(),
        ]

        # Q-values for option-value function
        self.option_value_net = nn.Sequential(
            nn.Linear(state_dim + 3, 64),  # state + one-hot option
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(
            list(self.option_policy.parameters())
            + list(self.option_value_net.parameters()),
            lr=0.001,
        )

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99

    def act(
        self, state: np.ndarray, current_option: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Select action using current option or select new option"""
        if current_option is None or self.options[current_option].is_terminated(state):
            # Select new option
            current_option = self.option_policy.select_option(state)

        # Get action from current option's policy
        action = self.options[current_option].policy(state)

        return action, current_option

    def store_transition(self, state, option, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append(
            (state.copy(), option, action.copy(), reward, next_state.copy(), done)
        )

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, options, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states))
        options_t = torch.LongTensor(options)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)

        # Create one-hot encoding for options
        options_onehot = F.one_hot(options_t, num_classes=3).float()

        # Compute current option values
        option_inputs = torch.cat([states_t, options_onehot], dim=1)
        current_values = self.option_value_net(option_inputs).squeeze()

        # Compute target option values
        with torch.no_grad():
            # Get Q-values for all options in next state
            next_option_values = []
            for i in range(3):
                option_onehot = F.one_hot(
                    torch.tensor([i] * self.batch_size), num_classes=3
                ).float()
                next_option_inputs = torch.cat([next_states_t, option_onehot], dim=1)
                next_values = self.option_value_net(next_option_inputs)
                next_option_values.append(next_values)

            next_option_values = torch.stack(next_option_values, dim=1).squeeze()

            # Get option selection probabilities
            option_probs = F.softmax(self.option_policy(next_states_t), dim=1)

            # Compute expected value
            expected_values = (option_probs * next_option_values).sum(dim=1)

            # Compute targets
            targets = rewards_t + self.gamma * (1 - dones_t) * expected_values

        # Compute loss
        loss = F.mse_loss(current_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_lunar_lander(
    options_agent: OptionCriticAgent, env: gym.Env, episodes: int = 1000
):
    """Train the agent using options framework"""

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        current_option = None
        option_durations = []
        current_option_start = 0

        while not done and step < 1000:
            # Select action using current option
            action, new_option = options_agent.act(state, current_option)

            # Check if option changed
            if current_option != new_option:
                if current_option is not None:
                    option_durations.append(step - current_option_start)
                current_option = new_option
                current_option_start = step

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Store transition
            options_agent.store_transition(
                state, current_option, action, reward, next_state, done
            )

            # Train
            loss = options_agent.train_step()

            state = next_state
            total_reward += reward
            step += 1

            if done:
                if current_option is not None:
                    option_durations.append(step - current_option_start)
                break

        episode_rewards.append(total_reward)

        # Print progress
        if episode % 50 == 0:
            avg_reward = (
                np.mean(episode_rewards[-50:])
                if episode >= 50
                else np.mean(episode_rewards)
            )
            print(
                f"Episode {episode}, Reward: {total_reward:.2f}, "
                f"Avg Reward (last 50): {avg_reward:.2f}, "
                f"Option durations: {option_durations}"
            )

    return episode_rewards


def evaluate_policy(options_agent: OptionCriticAgent, env: gym.Env, episodes: int = 10):
    """Evaluate the trained policy"""
    print("\n" + "=" * 50)
    print("Evaluating Policy")
    print("=" * 50)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        current_option = None
        option_sequence = []

        while not done and step < 1000:
            action, new_option = options_agent.act(state, current_option)

            if current_option != new_option:
                if current_option is not None:
                    option_sequence.append((current_option, step))
                current_option = new_option

            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            total_reward += reward
            step += 1

            if done:
                if current_option is not None:
                    option_sequence.append((current_option, step))
                break

        print(f"Evaluation Episode {episode + 1}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Option Sequence: {option_sequence}")

        # Decode option names
        option_names = ["Flip", "Accelerate", "Decelerate"]
        decoded_sequence = [
            (option_names[opt], duration) for opt, duration in option_sequence
        ]
        print(f"  Option Names: {decoded_sequence}")
        print()


def main():
    """Main training and evaluation loop"""
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True)

    # Get state dimension
    state_dim = env.observation_space.shape[0]

    # Create agent
    agent = OptionCriticAgent(state_dim)

    # Train agent
    print("Training.")
    rewards = train_lunar_lander(agent, env, episodes=50000)

    # Evaluate
    evaluate_policy(agent, env, episodes=5)

    # Save the trained model
    torch.save(
        {
            "option_policy": agent.option_policy.state_dict(),
            "option_value_net": agent.option_value_net.state_dict(),
        },
        "lunar_lander_options.pth",
    )

    env.close()

    print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")


if __name__ == "__main__":
    main()
