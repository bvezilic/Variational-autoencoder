import torch
import torch.nn.functional as F


def loss_criterion(inputs, targets, logvar, mu):
    bce_loss = F.binary_cross_entropy(inputs, targets)
    kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce_loss + kl_loss
