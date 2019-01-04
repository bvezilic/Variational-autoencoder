import json

from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from loss import loss_criterion
from model import VAE
from utils import load_checkpoint, save_checkpoint
from config import *


class Trainer:
    def __init__(self, model, data_loader, optimizer, device):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device

    def run_train_loop(self, epochs):
        self.model.to(self.device)
        self.model.train()

        losses = []
        for epoch in range(1, epochs+1):
            print("-" * 40)
            print("Epoch {}".format(epoch))

            running_loss = 0
            for inputs, _ in self.data_loader:
                self.optimizer.zero_grad()

                x = inputs.view(inputs.shape[0], -1).to(self.device)
                y = (x > 0.5).float().to(self.device)
                y_hat, logvar, mu = self.model(x)

                loss = loss_criterion(y_hat, y, logvar, mu)
                loss.backward()
                self.optimizer.step()

                running_loss += loss

            epoch_loss = running_loss / len(self.data_loader)
            losses.append(epoch_loss)
            print("Loss: {:.4f}".format(epoch_loss))

        return losses


def train():
    # Initialize config file
    config = json.load(open(args.config, "r"))

    # Load data set
    dataset = MNIST(root="data", transform=ToTensor(), download=True)

    # Create data loader
    data_loader = DataLoader(dataset=dataset, batch_size=config["batch_size"])

    # Load from checkpoint if passed
    if args.resume:
        model, optimizer = load_checkpoint(os.path.join(MODELS_ROOT, args.resume))
    else:
        # Initialize model
        model = VAE(**config["model_params"])

        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=config["lr"])

    # Initialize trainer
    trainer = Trainer(model=model,
                      data_loader=data_loader,
                      optimizer=optimizer,
                      device=args.device)

    # Run training
    trainer.run_train_loop(config["epochs"])

    # Save model
    save_checkpoint(model, optimizer, path=os.path.join(MODELS_ROOT, config["save_path"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device on which to run training")
    parser.add_argument("--resume", default=None,
                        help="Path to .pt model")
    parser.add_argument("--config", default="config.json",
                        help="Path to config file")

    args = parser.parse_args()
    train()
