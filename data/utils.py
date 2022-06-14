import torch

from math import sqrt

def add_gaussian_noise(tensor, mean=0., std=1.):
    return tensor + torch.randn(tensor.size()) * std + mean