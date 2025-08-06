from collections import deque
import os
import torch.nn as nn
from models import Actor, Critic, Discriminator
import time
import random
import numpy as np
import torch
import gymnasium as gym
from isaac_utils import DiscriminatorBufferGroup

class Agent(nn.Module):
    def __init__(self, envs):
        self.critic = Critic(state_dim=np.array(envs.observation_space.shape).prod(), value_dim=1, hidden_size=256)
        self.actor = Actor(state_dim=np.array(envs.observation_space.shape).prod(), action_dim=np.array(envs.action_space.shape).prod(), hidden_size=256, init_log_std=0.05)
    
    def get_value(self, state):
        value = self.critic(state)
        return value
    
    def get_action_and_value(self, state):
        dist = self.actor(state)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy().sum(dim=1), self.critic(state)

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
        motion_file: str = "assets/motions/run.json",
        num_discriminators: int = 32,
        discriminator_lr: float = 0.001,
        lambda_gp: float = 10,
        sequence_length: int = 5,
    ):
        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_iterations = total_timesteps // (self.batch_size)
        self.run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
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
            render_mode="rgb_array",
            num_envs=num_envs,
            motion_file=motion_file
        )

        video_kwargs = {
            "video_folder": os.path.join(self.log_root_path, "videos", "train"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)

        self.agent = Agent(self.env)
        self.sequence_length = sequence_length
        self.discriminator_buffer_group = DiscriminatorBufferGroup(count=num_envs, capacity=10000, sequence_length=sequence_length)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.discriminator = Discriminator(
            state_dim=np.array(self.env.single_observation_space.shape).prod(), 
            num_discriminators=num_discriminators
        )

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, eps=1e-5)
        self.lambda_gp = lambda_gp

        self.minibuffer = deque(maxlen=5)
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
        self.minibuffer.append(next_obs)
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
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
                if len(self.minibuffer) < self.sequence_length:
                    imitation_reward = 0
                else:
                    imitation_reward = torch.clip(self.discriminator(torch.stack(self.minibuffer).unsqueeze(0)), -1, 1).mean()
                
                self.minibuffer.append(next_obs)
                reward += imitation_reward

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
                
                self.discriminator_buffer_group.clear_minibuffer(next_done)
                self.train_discriminator(next_done)
                reset_indices = torch.nonzero(next_done, as_tuple=True)[0]
                self.env.unwrapped._reset_idx(reset_indices)

                with torch.no_grad():
                    next_value = self.agent.get_value(next_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                b_inds = np.arange(self.batch_size)
                clipfracs = []
                for epoch in range(self.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, self.batch_size, self.minibatch_size):
                        end = start + self.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if self.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -self.clip_coef,
                                self.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                        optimizer.step()

                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break
        self.env.close()
    def train_discriminator(self, next_done):
        policy_data = self.discriminator_buffer_group.buffers.sample(next_done)
        expert_data = self.env.reference_motion.sample(64, sample_length=5)

        alpha = torch.rand(expert_data.size(0), dtype=expert_data.dtype, device=expert_data.device)
        interpolated_data = alpha * expert_data + (1 - alpha) * policy_data
        interpolated_data.requires_grad = True

        dout = self.discriminator(interpolated_data)
        gradient = torch.autograd.grad(
            dout, interpolated_data, torch.ones_like(dout),
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]
        
        gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).sum()
        zero_tensor = torch.zeros_like(policy_data)
        one_tensor = torch.ones_like(policy_data)
        hinge_loss = torch.maximum(zero_tensor, one_tensor + self.discriminator(policy_data).sum() 
            + torch.maximum(zero_tensor, one_tensor-self.discriminator(expert_data).sum()))

        total_loss = (hinge_loss + self.lambda_gp * gradient_penalty) / self.num_discriminators
        self.discriminator_optimizer.zero_grad()
        total_loss.backward()
        self.discriminator_optimizer.step()
