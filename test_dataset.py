'''
Test Suite and Debugging for ./data/*
'''

from datafiles.loaders import dset2loader
from skew import quantity_skew, feature_skew
import numpy as np

names = ['mnist', 'kmnist', 'svhn', 'cifar10']

def test_load():
    for name in names:
        print(f"Testing: loader of [{name}]...")
        tr_l, te_l = dset2loader(name)
        for i, (x, y) in enumerate(tr_l):
            if i != 0:
                break
            print(x.size())
            print(y.size())
        for i, (x, y) in enumerate(te_l):
            if i != 0:
                break
            print(x.size())
            print(y.size())
        print("..Ok")

def test_qskew():
    for name in names:
        print(f"Testing: quantity skewing [{name}]")
        quantity_skew(name, 10)
        print("..Ok")

if __name__ == "__main__":
    np.random.seed(400)
    test_qskew()