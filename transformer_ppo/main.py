import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import random
import os
import time
from collections import deque

def init_layer(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

def make_env(env_id, idx, capture_video, video_folder, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=200)
            env = gym.wrappers.RecordVideo(env, video_folder)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class PositionalEncoding(nn.Module):
    def __init__(self, latent_size, length, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(length, latent_size) # (length, latent_size)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1) # (length, 1)
        div_term = torch.exp(
            torch.arange(0, latent_size, 2) * (-math.log(10000.0) / latent_size)
        ) # (latent_size / 2)

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(0) # (1, length, latent_size)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, embedding):
        embedding = embedding + self.positional_encoding[:embedding.size(0)]
        return self.dropout(embedding)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, context_length=8, latent_dim=256, heads=8):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Each Observation is (s1, a1, s2, a2, s3, ..., s8, a8, s9)
        self.obs_embedding = init_layer(nn.Linear(state_dim, latent_dim))
        self.next_obs_embedding = init_layer(nn.Linear(state_dim, latent_dim))
        self.action_embedding = init_layer(nn.Linear(action_dim, latent_dim))

        self.positional_encoding = PositionalEncoding(latent_dim, context_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim*3, nhead=heads, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=8
        )

        self.action_mean = init_layer(nn.Linear(latent_dim*3, action_dim))
        self.action_log_std = init_layer(nn.Linear(latent_dim*3, action_dim))

    def forward(self, states, actions, next_states):
        # states: s1 to s8 --> (B, seq_len, state_dim)
        # actions: a1 to a8 --> (B, seq_len, action_dim)
        # next_states: s2 to s9 --> (B, seq_len, state_dim)
        B, seq_len, _ = states.shape
        padding_mask = torch.where(states == 0, 1, 0).any(dim=-1) # (B, seq_len)

        states = states.reshape(-1, self.state_dim) # (B * seq_len, state_dim)
        next_states = next_states.reshape(-1, self.state_dim) # (B * seq_len, state_dim)
        actions = actions.reshape(-1, self.action_dim) # (B * seq_len, action_dim)

        # (B * seq_len, latent_dim)
        state_embedding = self.obs_embedding(states) 
        action_embedding = self.action_embedding(actions)
        next_state_embedding = self.next_obs_embedding(next_states)

        # (B, seq_len, latent_dim)  
        state_embedding = state_embedding.view(B, seq_len, self.latent_dim) 
        action_embedding = action_embedding.view(B, seq_len, self.latent_dim)
        next_state_embedding = next_state_embedding.view(B, seq_len, self.latent_dim)

        # Positional Encoding: (B, seq_len, latent_dim)
        state_embedding = self.positional_encoding(state_embedding)
        action_embedding = self.positional_encoding(action_embedding)
        next_state_embedding = self.positional_encoding(next_state_embedding)
        combined_embedding = torch.cat([state_embedding, action_embedding, next_state_embedding], dim=-1) # (B, seq_len, latent_dim*3)
        transformer_out = self.transformer_encoder(
            combined_embedding, 
            src_key_padding_mask=padding_mask
        ) # (B, seq_len, latent_dim*3)

        last_token = transformer_out[:, -1, :] # (B, latent_dim*3)
        action_mean = self.action_mean(last_token)
        action_log_std = self.action_log_std(last_token)
        return action_mean, torch.clamp(action_log_std, -10, 10).exp()

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            init_layer(nn.Linear(np.array(state_dim).prod(), 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, state):
        return self.critic(state)

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, context_length=8):
        super(Agent, self).__init__()
        self.actor = Actor(state_dim, action_dim, context_length=context_length)
        self.critic = Critic(state_dim)

    def get_action_and_value(self, states, actions, next_states):
        # States: (B, seq_len, state_dim)
        action_mean, action_std = self.actor(states, actions, next_states)
        distribution = Normal(loc=action_mean, scale=action_std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        current_state = states[:, -1, :] # (B, 1, state_dim)
        current_state = current_state.squeeze(1)
        value = self.critic(current_state)
        return action, value, log_prob.sum(1), distribution.entropy().sum(1)

    def get_value(self, states):
        current_state = states[:, -1, :] # (B, 1, state_dim)
        current_state = current_state.squeeze(1)
        value = self.critic(current_state)
        return value

class TransformerPPO:
    def __init__(self,
        total_timesteps: int = 30000000,
        env_name = "MountainCarContinuous-v0",
        learning_rate: float = 0.0026,
        num_envs: int = 4,
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
        seed: int = 0,
        torch_deterministic: bool = False,
        context_length=8,
        video_interval=1000,
        record_video = True
    ):
        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches
        self.num_iterations = total_timesteps // (self.batch_size)
        self.run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_root_path = os.path.join("logs", self.run_name)
        self.log_root_path = os.path.abspath(self.log_root_path)
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.num_envs = num_envs

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        self.device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

        self.env = gym.vector.SyncVectorEnv(
            [make_env(env_name, i, record_video, 
            os.path.join(self.log_root_path, "videos"), gamma) for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        self.writer = SummaryWriter(os.path.join(self.log_root_path, "data"))

        self.agent = Agent(
            state_dim=np.prod(self.env.single_observation_space.shape),
            action_dim=np.prod(self.env.single_action_space.shape),
            context_length=context_length
        )
        self.agent.to(self.device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        self.learning_rate = learning_rate
        self.global_step = 0

        self.state_minibuffer = deque(
            [
                torch.zeros((
                    self.num_envs, 
                    np.prod(self.env.single_observation_space.shape)), device=self.device) for i in range(context_length)
            ], 
            maxlen=context_length
        )
        self.action_minibuffer = deque(
            [
                torch.zeros((
                    self.num_envs, 
                    np.prod(self.env.single_action_space.shape)), device=self.device) for i in range(context_length)
            ], 
            maxlen=context_length
        )
        self.next_state_minibuffer = deque(
            [
                torch.zeros((
                    self.num_envs, 
                    np.prod(self.env.single_observation_space.shape)), device=self.device) for i in range(context_length)
            ], 
            maxlen=context_length
        )

        self.context_length = context_length
        self.episodic_reward = np.array([0 for i in range(self.num_envs)], dtype=np.float32)

    
    def train(self):
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        obs = torch.zeros((self.num_steps, self.num_envs, self.env.observation_space.shape[-1]), dtype=torch.float).to(self.device)
        temporal_obs = torch.zeros((self.num_steps, self.num_envs, self.context_length, np.prod(self.env.single_observation_space.shape)), dtype=torch.float).to(self.device)
        temporal_next_obs = torch.zeros((self.num_steps, self.num_envs, self.context_length, np.prod(self.env.single_observation_space.shape)), dtype=torch.float).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs, self.env.action_space.shape[-1]), dtype=torch.float).to(self.device)
        temporal_actions = torch.zeros((self.num_steps, self.num_envs, self.context_length, np.prod(self.env.single_action_space.shape)), dtype=torch.float).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float).to(self.device)
        advantages = torch.zeros_like(rewards, dtype=torch.float).to(self.device)

        next_obs, _ = self.env.reset()        
        self.state_minibuffer.append(torch.tensor(next_obs).to(self.device))
        next_done = torch.zeros(self.num_envs, dtype=torch.float).to(self.device)

        for iteration in range(1, self.num_iterations+1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lr_now = self.learning_rate * frac
                optimizer.param_groups[0]["lr"] = lr_now
            
            for step in range(0, self.num_steps):
                self.global_step += self.num_envs
                obs[step] = torch.tensor(next_obs)
                dones[step] = next_done

                with torch.no_grad():
                    current_temporal_obs = torch.stack(list(self.state_minibuffer)).to(self.device).permute(1, 0, 2)
                    current_temporal_next_obs = torch.stack(list(self.next_state_minibuffer)).to(self.device).permute(1, 0, 2)
                    current_temporal_action = torch.stack(list(self.action_minibuffer)).to(self.device).permute(1, 0, 2)

                    temporal_obs[step] = current_temporal_obs
                    temporal_next_obs[step] = current_temporal_next_obs
                    temporal_actions[step] = current_temporal_action

                    action, value, logprob, _ = self.agent.get_action_and_value(current_temporal_obs, current_temporal_action, current_temporal_next_obs)
                    values[step] = value.flatten().to(self.device)

                actions[step] = action
                logprobs[step] = logprob
                
                next_obs, reward, terminations, truncations, infos = self.env.step(action.cpu())                
                
                self.state_minibuffer.append(torch.tensor(next_obs).to(device=self.device))
                self.action_minibuffer.append(torch.tensor(action).to(device=self.device))
                self.next_state_minibuffer.append(torch.tensor(next_obs).to(device=self.device))
                self.episodic_reward += reward
                reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, device=self.device).view(-1)
                next_obs, next_done = torch.tensor(next_obs, device=self.device), torch.tensor(next_done, device=self.device, dtype=torch.float32)
                
                reset_indices = torch.nonzero(next_done, as_tuple=True)[0]
                for i in reset_indices:
                    for ctx_state, ctx_action, ctx_next_state in zip(self.state_minibuffer, self.action_minibuffer, self.next_state_minibuffer):
                        ctx_state[i] = torch.zeros(np.prod(self.env.single_observation_space.shape))
                        ctx_action[i] = torch.zeros(np.prod(self.env.single_action_space.shape))
                        ctx_next_state[i] = torch.zeros(np.prod(self.env.single_observation_space.shape))

                        self.writer.append("evaluation/episodic_reward", torch.tensor(
                            self.episodic_reward
                        )[reset_indices].mean())

                        self.episodic_reward[i] = 0
                        self.env.num_envs[i].reset()

                with torch.no_grad():
                    
                    next_value = self.agent.get_value(current_temporal_obs).reshape(1, -1)
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = (1.0 - next_done).to(self.device)
                            nextvalues = next_value.to(self.device)
                        else:
                            nextnonterminal = (1.0 - dones[t + 1]).to(self.device)
                            nextvalues = values[t + 1].to(self.device)
                        delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                b_temporal_obs = temporal_obs.reshape((-1, self.context_length, self.env.observation_space.shape[-1]))
                b_logprobs = logprobs.reshape(-1)
                b_actions = temporal_actions.reshape((-1, self.context_length, self.env.action_space.shape[-1]))
                b_temporal_next_obs = temporal_next_obs.reshape((-1, self.context_length, self.env.observation_space.shape[-1]))
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

                        _, newvalue, newlogprob, entropy = self.agent.get_action_and_value(
                            b_temporal_obs[mb_inds], b_actions[mb_inds], b_temporal_next_obs[mb_inds]
                        )
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
                        self.writer.add_scalar("ppo/policy_loss", pg_loss.item(), self.global_step + epoch)

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

                        self.writer.add_scalar("ppo/value_loss", v_loss.item(), self.global_step + epoch)
                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                        self.writer.add_scalar("ppo/loss", loss.item(), self.global_step + epoch)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                        optimizer.step()

                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break
        self.env.close()

if __name__ == "__main__":
    ppo = TransformerPPO()
    ppo.train()