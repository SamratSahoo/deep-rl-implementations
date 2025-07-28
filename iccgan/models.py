import torch
import torch.nn as nn


class ActorCritic(nn.Module):

    class Critic(nn.Module):
        def __init__(self, state_dim, goal_dim, value_dim=1, latent_dim=256):
            super().__init__()
            self.gru = nn.GRU(
                input_size=state_dim, hidden_size=latent_dim, batch_first=True
            )

            self.mlp = nn.Sequential(
                nn.Linear(latent_dim+goal_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, value_dim)
            )

            for name, param in self.mlp.named_parameters():
                if "weight" in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.05)
                elif "bias" in name:
                    torch.nn.init.constant_(param, 0)
        
    class Actor(nn.Module):
        pass

class Discriminator(nn.Module):
    pass
