# Variational-autoencoder

This is another PyTorch implementation of Variational Autoencoder (VAE) trained MNIST dataset. The goal of this exercise is to get more familiar with older generative models such as family of autoencoders.

### Model

Upon reading the original paper, examples and tutorials for few days the whole model can be described in one image...



Input, encoder, decoder, output are all pretty straight-forward concepts, however what is exactly happening in the 
middle section with the *latent variable*?

Code-wise from [model.py](https://github.com/bvezilic/Variational-autoencoder/blob/master/model.py):

```python
class LatentZ(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, p_x):
        mu = self.mu(p_x)
        logvar = self.logvar(p_x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, logvar, mu
```

`mu` and `logvar` are standard fully connected layers that will represent mean and log-variance, respectfully.
Using outputs of these layers will be then used to sample latent variable `z`. A couple of things before we
can sample `z`, (1) compute `std` from `logvar`, (2) sample from normal distribution to get `eps`. Once `std` and `eps`
are obtained `z` can be computed as `std * eps + mu`.

A couple of differences compared to the original paper, *sigmoid* activations are replaced by *relu*. And instead of 
*SGD*, *Adam* optimizer was used (better default choice).

### Loss

The loss function consists of two terms. Reconstruction loss, for which was used `binary_cross_entropy`, in paper was used
*mean squared error*. And regularization loss or Kullback–Leibler divergence that will force `z` to be normal distribution with *mean=0* and *std=1*.

```python
def loss_criterion(inputs, targets, logvar, mu):
    # Reconstruction loss
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction="sum")
    # Regularization term
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce_loss + kl_loss
```

If `mu` and `logvar` were singular values, instead of vectors, plotting a 3D graph would look something like this:

 

### Issues

One interesting thing that happen in my initial attempts is that I would get these kind of results no matter how long
the training persisted.



See the `reduction` parameter in `binary_cross_entropy` function? Well, if that parameter was left to its default value 
of averaging loss per batch, the loss would always stay the same. However, when changed to *sum*, the reconstructions 
improved a lot.


### Resources
Original paper:
* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

Helpful GitHub repositories:

* https://github.com/wiseodd/generative-models
* https://github.com/bhpfelix/Variational-Autoencoder-PyTorch

Tutorial:
* https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
