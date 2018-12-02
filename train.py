import json

from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from loss import loss_criterion
from model import VAE


class Trainer:
    def __init__(self, model, data_loader, optimizer, device):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device

        self.model.to(device)

    def train(self, epochs):
        for epoch in range(1, epochs+1):
            print("Epoch {}".format(epoch))

            running_loss = 0
            for input, _ in tqdm(self.data_loader):
                self.optimizer.zero_grad()

                x = input.view(input.shape[0], -1).to(self.device)
                y = (x > 0.5).float().to(self.device)
                y_hat, logvar, mu = self.model(x)

                loss = loss_criterion(y_hat, y, logvar, mu)
                loss.backward()

                running_loss += loss

            print("Loss: {}\n\n".format(running_loss))

    def save_model(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model": {
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_sizze,
                "latent_size": self.model.latent_size
            }
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)

        self.model = VAE(**checkpoint["model"])
        self.optimizer = Adam(self.model.parameters())

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train(args):
    # Initialize config file
    config = json.load(open(args.config, "r"))

    # Load data set
    dataset = MNIST(root="data", transform=ToTensor())

    # Create data loader
    data_loader = DataLoader(dataset=dataset, batch_size=config["batch_size"])

    # Initialize model
    model = VAE(input_size=784, hidden_size=512, latent_size=20)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Initialize trainer
    trainer = Trainer(model=model,
                      data_loader=data_loader,
                      optimizer=optimizer,
                      device=args.device)
    trainer.train(config["epochs"])

    # Save model
    trainer.save_model(path=config.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device on which to run training")
    parser.add_argument("--resume", default="path_to_model",
                        help="Path to .pt model")
    parser.add_argument("--config", default="config.json",
                        help="Path to config file")

    args = parser.parse_args()
    train(args)
