import json

from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from vae.callbacks import PlotCallback
from vae.loss import loss_criterion
from vae.model import VAE
from vae.utils import load_checkpoint, save_checkpoint
from config import *


class Trainer:
    def __init__(self, model, data_loader, optimizer, device, callbacks=[]):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks

    def run_train_loop(self, epochs):
        self.model.to(self.device)  # Set model params to cpu/gpu
        self.model.train()  # Set model to train mode

        losses = []
        for epoch in range(1, epochs+1):
            print("-" * 20)
            print("Epoch {}".format(epoch))

            running_loss = 0
            for inputs, _ in self.data_loader:
                self.optimizer.zero_grad()

                # Prepare inputs and targets
                x = inputs.view(inputs.size(0), -1).to(self.device)
                y = (x > 0.5).float().to(self.device)

                # Forward pass
                y_hat, logvar, mu, _ = self.model(x)

                # Compute loss
                loss = loss_criterion(y_hat, y, logvar, mu)

                # Compute gradients and update weights
                loss.backward()
                self.optimizer.step()

                running_loss += loss

            epoch_loss = running_loss / len(self.data_loader)
            losses.append(epoch_loss.item())
            print("Loss: {:.4f}".format(epoch_loss))

            # On end of epoch call any callbacks
            if self.callbacks:
                [fn(self) for fn in self.callbacks]

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
        model, optimizer = load_checkpoint(os.path.join(MODELS_DIR, args.resume))
    else:
        # Initialize model
        model = VAE(**config["model_params"])

        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=config["lr"])

    # Initialize trainer
    trainer = Trainer(model=model,
                      data_loader=data_loader,
                      optimizer=optimizer,
                      device=args.device,
                      callbacks=[PlotCallback(save_dir=PROJECT_ROOT + "/images")])

    # Run training
    trainer.run_train_loop(config["epochs"])

    # Save model
    save_checkpoint(model, optimizer, path=os.path.join(MODELS_DIR, config["save_path"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device on which to run training")
    parser.add_argument("--resume", default=None,
                        help="Path to .pt model")
    parser.add_argument("--config", default="train_config.json",
                        help="Path to config file")

    args = parser.parse_args()
    train()
