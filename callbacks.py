import numpy as np
import torch
import matplotlib.pyplot as plt


class PlotSamples:
    def __init__(self, num_samples=4, save_dir=None):
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.counter = 0

    def __call__(self, trainer):
        trainer.model.eval()
        with torch.no_grad():
            inputs = self._random_samples(trainer)
            outputs, _, _, z = trainer.model(inputs)

            input_images = self._convert_to_image(inputs)
            recon_images = self._convert_to_image(outputs)
            z_numpy = z.cpu().numpy()

            self._plot_samples(input_images, recon_images, z_numpy)

        # Return to train mode
        trainer.model.train()

    def _random_samples(self, trainer):
        dataset = trainer.data_loader.dataset
        ids = np.random.randint(len(dataset), size=self.num_samples)

        samples = [dataset[idx][0] for idx in ids]
        samples = torch.stack(samples)
        samples = samples.view(samples.size(0), -1)
        samples = samples.to(trainer.device)

        return samples

    def _convert_to_image(self, tensor):
        images = tensor.reshape(-1, 28, 28)
        images = images.cpu().numpy()

        return images

    def _plot_samples(self, input_images, recon_images, z):
        fig, ax_lst = plt.subplots(self.num_samples, 3)
        fig.suptitle("Input -> Latent Z -> Reconstructed")

        for i in range(self.num_samples):
            # Images
            ax_lst[i][0].imshow(input_images[i], cmap="gray")
            ax_lst[i][0].set_axis_off()

            # Variable z
            ax_lst[i][1].bar(np.arange(len(z[i])), z[i])

            # Reconstructed images
            ax_lst[i][2].imshow(recon_images[i], cmap="gray")
            ax_lst[i][0].set_axis_off()

        if self.save_dir:
            fig.savefig(self.save_dir + "/results_{}.png".format(self.counter))
            plt.close(fig)
            self.counter += 1
        else:
            plt.show(fig)


