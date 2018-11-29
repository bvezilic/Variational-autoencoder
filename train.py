import json

from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST

from models import VAE


class Trainer:
    def __init__(self, model, data_loader, optimizer):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer

    def train(self, epochs):
        for epoch in range(epochs):
            print("-" * 20)
            loss = self._run_epoch()

    def _run_epoch(self):
        running_loss = 0
        for input, target in self.data_loader:

            self.optimizer.zero_grad()


def train(args):
    # Initialize config file
    config = json.load(open(args.config, "r"))
    train_config = config["trainer"]

    # Load data set
    dataset = MNIST(root="data",
                    transform=None)

    # Create data loader
    data_loader = DataLoader(dataset=dataset,
                             batch_size=train_config["batch_size"])

    # Initialize model
    model = VAE()

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=train_config["lr"])

    # Initialize trainer
    trainer = Trainer(model=model,
                      data_loader=data_loader,
                      optimizer=optimizer)

    trainer.train()

    train_config.save_model(path=config.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default="path_to_model",
                        help="Path to ")
    parser.add_argument("--config",
                        help="Path to config file")

    args = parser.parse_args()
    train(args)
