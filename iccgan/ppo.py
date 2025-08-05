import os
import torch.nn as nn
from models import Actor, Critic
import time
import random
import numpy as np
import torch
import gymnasium as gym
from isaac_utils import DiscriminatorBuffer

class Agent(nn.Module):
    def __init__(self, envs):
        self.critic = Critic(state_dim=envs.observation_space.shape[-1], value_dim=1, hidden_size=256)
        self.actor = Actor(state_dim=envs.observation_space.shape[-1], action_dim=envs.action_space.shape[-1], hidden_size=256, init_log_std=0.05)
    
    def get_value(self, state, seq_length):
        value = self.critic(state, seq_length)
        return value
    
    def get_action_and_value(self, state, seq_length):
        dist = self.actor(state, seq_length)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy().sum(dim=1), self.critic(state, seq_length)

class PPO:
    def __init__(
        self,
        total_timesteps: int = 30000000,
        learning_rate: float = 0.0026,
        num_envs: int = 4096,
        num_steps: int = 16,
        anneal_lr: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 2,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        ent_coef: float = 0.0,
        vf_coef: float = 2,
        max_grad_norm: float = 1,
        target_kl: float = None,
        reward_scaler: float = 1,
        video_interval: int = 1464,
        video_length: int = 500,
        seed: int = 0,
        torch_deterministic: bool = False,
    ):
        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_iterations = total_timesteps // (self.batch_size)
        self.run_name = f"{int(time.time())}"
        self.log_root_path = os.path.join("logs", self.run_name)
        self.log_root_path = os.path.abspath(self.log_root_path)
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(
            "Isaac-ICCGAN-v0", 
            disable_env_checker=True, 
            render_mode="rgb_array"
        )

        video_kwargs = {
            "video_folder": os.path.join(self.log_root_path, time.strftime("%Y-%m-%d_%H-%M-%S"), "videos", "train"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)

        self.agent = Agent(self.env)
        self.discriminator_buffer = DiscriminatorBuffer(capacity=10000)
    
    def train(self):
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        obs = torch.zeros((self.num_steps, self.num_envs) + self.env.single_observation_space.shape, dtype=torch.float).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + self.env.single_action_space.shape, dtype=torch.float).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        advantages = torch.zeros_like(rewards, dtype=torch.float).to(self.device)
        
        global_step = 0
        next_obs = self.env.reset()
        next_done = torch.zeros(self.num_envs, dtype=torch.float).to(self.device)

        for iteration in range(1, self.num_iterations+1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lr_now = self.learning_rate * frac
                optimizer.param_groups[0]["lr"] = lr_now
            
            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs, self.num_envs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob