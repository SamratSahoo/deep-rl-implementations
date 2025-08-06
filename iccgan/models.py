import torch
import torch.nn as nn



class Critic(nn.Module):
    def __init__(self, state_dim, value_dim=1, hidden_size=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim, hidden_size=hidden_size, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
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
    
    def forward(self, sequence):
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        out, hidden = self.gru(sequence)

        out = out[:, -1, :]
        out = self.mlp(out)
        return out
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, init_log_std=0.05):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim, hidden_size=hidden_size, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.policy_mu = nn.Linear(512, action_dim)
        self.policy_log_sigma = nn.Linear(512, action_dim)

        torch.nn.init.constant_(self.policy_log_sigma.weight, 0.)
        torch.nn.init.constant_(self.policy_log_sigma.bias, init_log_std)

        for name, param in self.mlp.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.05)
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)

        # for name, param in self.gru.named_parameters():
        #     if 'weight_ih' in name:
        #         torch.nn.init.xavier_uniform_(param.data)
        #     elif 'weight_hh' in name:
        #         torch.nn.init.orthogonal_(param.data)
        #     elif 'bias' in name:
        #         param.data.fill_(0)

    
    def forward(self, sequence):
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        
        out, hidden = self.gru(sequence)
        out = out[:, -1, :]
        out = self.mlp(out)
        mu = self.policy_mu(out)
        log_sigma = self.policy_log_sigma(out)

        dist = torch.distributions.normal.Normal(loc=mu, scale=log_sigma.exp())
        return dist

class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_size=256, num_discriminators=8):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim, hidden_size=hidden_size, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_discriminators)
        )

        i = 0
        for name, param in self.mlp.named_parameters():
            if "weight" in name:
                gain = 1 if i == 2 else 2**0.5 
                torch.nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)
            i += 1
    
    def forward(self, sequence):
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        out, hidden = self.gru(sequence)
        out = out[:, -1, :]
        out = self.mlp(out)
        return out




