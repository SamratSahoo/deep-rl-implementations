import numpy as np
from utils.replay_buffer import ReplayBuffer
import torch.nn as nn
import gymnasium as gym
import torch.optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim

        # Convolutional Layers
        self.conv1 = nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear Layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)

        x = F.softmax(x, dim=-1)
        return 2 * x - 1


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_network = nn.Sequential(
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
            nn.Linear(128, 128),
            nn.Softmax(dim=-1),
        )

        self.action_network = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )

        self.q_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, state, action):
        state_output = self.state_network(state)
        action_output = self.action_network(action)
        return self.q_network(torch.cat([state_output, action_output], dim=-1))


class DDPG:
    def __init__(
        self,
        environment: gym.Env,
        update_frequency=100,
        replay_buffer_capacity=10000,
        replay_buffer_batch_size=32,
        discount_factor=0.99,
        update_rate=0.01,
        num_updates=10,
        rollout_length=100,
        validation_episode_frequency=3,
    ):
        self.actor = Actor(
            state_dim=environment.observation_space.shape[0],
            action_dim=environment.action_space.shape[0],
        )
        self.critic = Critic(
            state_dim=environment.observation_space.shape[0],
            action_dim=environment.action_space.shape[0],
        )
        self.actor_target = Actor(
            state_dim=environment.observation_space.shape[0],
            action_dim=environment.action_space.shape[0],
        )
        self.critic_target = Critic(
            state_dim=environment.observation_space.shape[0],
            action_dim=environment.action_space.shape[0],
        )
        self.environment = environment
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        self.update_frequency = update_frequency
        self.replay_buffer_batch_size = replay_buffer_batch_size
        self.discount_factor = discount_factor
        self.update_rate = update_rate
        self.num_updates = num_updates
        self.rollout_length = rollout_length
        self.validation_episode_frequency = validation_episode_frequency

        self.critic_loss_fn = nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

    def update_networks(self):
        sampled_batch = self.replay_buffer.sample(self.replay_buffer_batch_size)
        state, action, reward, next_state, done = sampled_batch
        target_q = reward + (
            self.discount_factor
            * ~done
            * self.critic(next_state, self.actor(next_state))
        )
        critic_q = self.critic(state, action)

        # Make Critic Better based on TD error
        self.critic.zero_grad()
        critic_loss = self.critic_loss_fn(critic_q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Make actor better using better critic
        self.actor.zero_grad()
        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of parameters
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.update_rate * param.data
                + (1 - self.update_rate) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.update_rate * param.data
                + (1 - self.update_rate) * target_param.data
            )

    def validate(self, episodes=3):
        total_reward = 0
        for i in range(episodes):
            state, _ = self.environment.reset()
            done = False
            while not done:
                self.actor.eval()
                action = np.clip(
                    self.actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    .cpu()
                    .detach()
                    .numpy(),
                    self.environment.action_space.low,
                    self.environment.action_space.high,
                ).squeeze(0)
                self.actor.train()

                next_state, reward, done, _, _ = self.environment.step(action)
                total_reward += reward
                state = next_state
        return total_reward / episodes

    def train(self, episodes=100):
        state, _ = self.environment.reset()
        current_step = 0
        for i in range(episodes):
            done = False
            print(f"Training Episode {i + 1} / {episodes}")
            while not done:
                current_step += 1
                epsilon = np.random.normal(size=self.environment.action_space.shape[0])

                self.actor.eval()
                action = np.clip(
                    self.actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    .cpu()
                    .detach()
                    .numpy()
                    + epsilon,
                    self.environment.action_space.low,
                    self.environment.action_space.high,
                ).squeeze(0)
                self.actor.train()

                next_state, reward, done, _, info = self.environment.step(action)
                self.replay_buffer.append(
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(action, dtype=torch.float32),
                    torch.tensor(reward, dtype=torch.float32),
                    torch.tensor(next_state, dtype=torch.float32),
                    torch.tensor(done, dtype=torch.bool),
                )
                state = next_state
                if current_step < self.rollout_length:
                    continue

                if current_step % self.update_frequency == 0:
                    print("Updating Networks at Step", current_step)
                    for i in range(self.num_updates):
                        self.update_networks()

            if (i + 1) % self.validation_episode_frequency == 0:
                validation_reward = self.validate()
                print(f"Average Reward: {validation_reward}")
