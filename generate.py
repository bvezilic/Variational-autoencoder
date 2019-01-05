import os

import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal

from config import MODELS_ROOT
from utils import load_model


class Generator:
    def __init__(self, model):
        self.model = model

    def sample(self):
        with torch.no_grad():
            normal_dist = Normal(0, 1)
            randn_sample = normal_dist.sample((1, self.model.latent_size))

            sample = self.model.decoder(randn_sample)
            sample = sample.squeeze()
            sample = sample.reshape(28, 28)
            sample = sample.cpu().numpy()

            return sample


def generate():
    model = load_model(os.path.join(MODELS_ROOT, args.model_name))
    generator = Generator(model)

    sample = generator.sample()

    plt.imshow(sample)
    print("STOP")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="hs-512_ls-20_e-50.pt", help="Path to .pt model")

    args = parser.parse_args()
    generate()
