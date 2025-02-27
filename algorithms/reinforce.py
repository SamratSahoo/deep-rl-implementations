import numpy as np
from utils.replay_buffer import ReplayBuffer
import torch.nn as nn
import gymnasium as gym
import torch.optim
import torch.nn.functional as F
import torch.distributions as dist


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        # Takes a state, outputs means for a normal distribution for each action dimension
        self.mean_network = nn.Sequential(
            # Convolutional Layers
            nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Adaptive Pooling to get 1x1x512
            nn.AdaptiveAvgPool2d((1, 1)),
            # Linear Layers
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Takes a state, outputs std_dev for a normal distribution for each action dimension
        self.std_dev_network = nn.Sequential(
            # Convolutional Layers
            nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Adaptive Pooling to get 1x1x512
            nn.AdaptiveAvgPool2d((1, 1)),
            # Linear Layers
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.mean_network(state), self.std_dev_network(state) + 1e-6


class Reinforce:
    def __init__(self, environment, lr=0.01, gamma=0.99, max_episode_len=1000):
        self.max_episode_len = max_episode_len
        self.policy_network = PolicyNetwork(
            environment.observation_space.shape[0], environment.action_space.shape[0]
        )
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.environment = environment

    def get_discounted_rewards(self, trajectory):
        rewards = [state[2] for state in trajectory]
        discounted_rewards = []
        for i in range(len(rewards)):
            discounted_reward = 0
            for j in range(i, len(rewards)):
                discounted_reward += rewards[j] * (self.gamma ** (j - i))
            discounted_rewards.append(discounted_reward)
        return discounted_rewards

    def choose_action(self, state):
        self.policy_network.eval()
        mean, std_dev = self.policy_network(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        )
        self.policy_network.train()

        mean = mean.squeeze(0)
        std_dev = std_dev.squeeze(0)
        distribution = torch.distributions.Normal(mean, std_dev)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)
        action = np.clip(
            action.cpu().detach().numpy(),
            self.environment.action_space.low,
            self.environment.action_space.high,
        )
        return action, log_prob

    def update_network(self, trajectory, log_probs):
        returns = self.get_discounted_rewards(trajectory)
        loss = -torch.sum(log_probs * returns)
        print("Loss: ", loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=100):
        for i in range(episodes):
            print(f"Training Episode {i + 1} / {episodes}")
            done = False
            state, _ = self.environment.reset()
            current_trajectory = []
            current_log_probs = []
            current_step = 0

            while not done and current_step < self.max_episode_len:
                print(current_step)
                action, log_probs = self.choose_action(state)
                current_log_probs.append(log_probs)
                next_state, reward, done, _, _ = self.environment.step(action)

                current_trajectory.append(
                    (
                        torch.tensor(state, dtype=torch.float32),
                        torch.tensor(action, dtype=torch.float32),
                        torch.tensor(reward, dtype=torch.float32),
                        torch.tensor(next_state, dtype=torch.float32),
                        torch.tensor(done, dtype=torch.bool),
                    )
                )
                state = next_state
                current_step += 1
            self.update_network(current_trajectory, log_probs)
