import torch
import torch.nn.functional as F


def loss_criterion(input, target, logvar, mu):
    bce_loss = F.binary_cross_entropy(input, target)
    kl_loss = 0.5 * torch.sum(1 + mu.pow(2) - logvar.exp())

    return bce_loss + kl_loss
