import torch
import numpy as np

def add_gaussian_noise(tensor, mean=0., std=1.):
    return tensor + torch.randn(tensor.size()) * std + mean

def setseed(seed):
    np.random.seed(seed)