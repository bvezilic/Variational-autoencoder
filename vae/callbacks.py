import os
import numpy as np
import torch
import matplotlib.pyplot as plt


class PlotCallback:
    """Callback class that retrieves several samples and displays model reconstructions"""
    def __init__(self, num_samples=4, save_dir=None):
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.counter = 0

        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, trainer):
        trainer.model.eval()  # Set model to eval mode due to Dropout, BN, etc.
        with torch.no_grad():
            inputs = self._batch_random_samples(trainer)
            outputs, _, _, z = trainer.model(inputs)  # Forward pass

            # Prepare data for plotting
            input_images = self._reshape_to_image(inputs, numpy=True)
            recon_images = self._reshape_to_image(outputs, numpy=True)
            z_ = self._to_numpy(z)

            self._plot_samples(input_images, recon_images, z_)

        trainer.model.train()  # Return to train mode

    def _batch_random_samples(self, trainer):
        """Helper function that retrieves `num_samles` from dataset and prepare them in batch for model """
        dataset = trainer.data_loader.dataset

        ids = np.random.randint(len(dataset), size=self.num_samples)
        samples = [dataset[idx][0] for idx in ids]  # Each data point is (image, class)

        # Create batch
        batch = torch.stack(samples)
        batch = batch.view(batch.size(0), -1)  # Flatten
        batch = batch.to(trainer.device)

        return batch

    def _reshape_to_image(self, tensor, numpy=True):
        """Helper function that converts image-vector into image-matrix."""
        images = tensor.reshape(-1, 28, 28)
        if numpy:
            images = self._to_numpy(images)

        return images

    def _to_numpy(self, tensor):
        """Helper function that converts tensor to numpy"""
        return tensor.cpu().numpy()

    def _plot_samples(self, input_images, recon_images, z):
        """Creates plot figure and saves it on disk if save_dir is passed."""
        fig, ax_lst = plt.subplots(self.num_samples, 3)
        fig.suptitle("Input → Latent Z → Reconstructed")

        for i in range(self.num_samples):
            # Images
            ax_lst[i][0].imshow(input_images[i], cmap="gray")
            ax_lst[i][0].set_axis_off()

            # Variable z
            ax_lst[i][1].bar(np.arange(len(z[i])), z[i])

            # Reconstructed images
            ax_lst[i][2].imshow(recon_images[i], cmap="gray")
            ax_lst[i][2].set_axis_off()

        fig.tight_layout()

        if self.save_dir:
            fig.savefig(self.save_dir + "/results_{}.png".format(self.counter))
            plt.close(fig)
            self.counter += 1
        else:
            plt.show(fig)
