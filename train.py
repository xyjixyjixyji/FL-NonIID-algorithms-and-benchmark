from numpy import require
from .models.resnet import *
from .models.digit import DigitModel
from .skew import quantity_skew, feature_skew_filter, feature_skew_noise, label_skew_across_labels, label_skew_by_within_labels
from .datafiles.loaders import dset2loader

import argparse

# COURTESY: TWO MODELS ARE FROM THE INTERNET, LINKS ATTACHED IN FILE
# when using svhn or cifar10, use resnet
# when using mnist or kmnist, use digitmodel

parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help ='whether to make a log')
parser.add_argument('--test', action='store_true', help ='test the pretrained model')
parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--batch', type = int, default= 32, help ='batch size')
parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedprox | fedbn')
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')

# skewing parameters
parser.add_argument('--dsetname', type=str, required=True)
parser.add_argument('--skew', type=str, required=True, help='what kind of skew you want')
parser.add_argument('--noise_std', type=float)
parser.add_argument('--filtersz', type=int)
parser.add_argument('--alpha', type=float, help='parameter of dirichlet distribution')

args = parser.parse_args()

dataset_name = args.dsetname
