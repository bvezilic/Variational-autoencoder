import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        p = F.relu(self.fc1(x))
        p = F.relu(self.fc2(p))

        return p


class LatentZ(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, logvar, mu


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, input_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        q = F.relu(self.fc1(z))
        q = F.sigmoid(self.fc2(q))

        return q


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = Encoder(input_size, hidden_size)
        self.latent_z = LatentZ(hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        p = self.encoder(x)
        z, logvar, mu = self.latent_z(p)
        q = self.decoder(z)

        return q, logvar, mu
