import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64)
        self.relu2 = nn.ReLU()

    def forward(self, samples):
        x = self.relu1(self.conv1(samples))
        logits = self.relu2(self.conv2(x))

        return logits


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 3)
        self.relu2 = nn.ReLU()

    def forward(self, *input):
        pass


class VAE(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = Encoder()
        self.z = torch.ones(z_dim)
        self.decoder = Decoder()

    def forward(self, *input):
        pass
