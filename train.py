from .models.resnet import *
from .models.digit import DigitModel
from .skew import quantity_skew, feature_skew_filter, feature_skew_noise, label_skew_across_labels, label_skew_by_within_labels
from .datafiles.loaders import dset2loader

import argparse
import torch
import torch.nn as nn

# COURTESY: TWO MODELS ARE FROM THE INTERNET, LINKS ATTACHED IN FILE
# when using svhn or cifar10, use resnet
# when using mnist or kmnist, use digitmodel

parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help='whether to make a log')
parser.add_argument('--test', action='store_true',
                    help='test the pretrained model')
parser.add_argument('--percent', type=float, default=0.1,
                    help='percentage of dataset to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--iters', type=int, default=100,
                    help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=1,
                    help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='fedbn',
                    help='fedavg | fedprox | fedbn')
parser.add_argument('--mu', type=float, default=1e-2,
                    help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type=str,
                    default='../checkpoint/digits', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true',
                    help='resume training from the save path checkpoint')

# skewing parameters
# TODO(for gyz/lyx): maybe make nclient(default 5 in fedbn code) to a changable parameter
parser.add_argument('--dsetname', type=str, required=True)
parser.add_argument('--skew', type=str, required=True,
                    help='| quantity | feat_filter | feat_noise | label_across | label_within |')
parser.add_argument('--noise_std', type=float)
parser.add_argument('--filtersz', type=int)
parser.add_argument('--alpha', type=float,
                    help='parameter of dirichlet distribution')

args = parser.parse_args()

# ==================== global parameters setup

dataset_name = args.dsetname
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#! Remember, model outputs prob distribution(10, ), but our target is NOT one-hot, it is (1,)
#! use torch.NLLLoss(prob_dist, target) to calculate instead of WASTING time on changing y to one-hot
model = None
if dataset_name in ['cifar10', 'svhn']:
    model = resnet20()
else:
    model = DigitModel(num_classes=10)


if __name__ == "__main__":
    pass
