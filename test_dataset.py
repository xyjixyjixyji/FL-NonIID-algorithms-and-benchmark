'''
Test Suite and Debugging for ./data/*
'''

from datafiles.loaders import dset2loader
from datafiles.utils import setseed
from datafiles.preprocess import preprocess
from skew import quantity_skew, feature_skew_noise, feature_skew_filter


names = ['mnist', 'kmnist', 'svhn', 'cifar10']
# mnist: 0 - 9
# kmnist: 0 - 9
# svhn: 0 - 9
# cifar10: 0 - 9

def test_load():
    for name in names:
        print(f"==================== Testing: loader of [{name}]...")
        tr_s, te_s = preprocess(dataset_name=name,
                                noise=False)
        tr_l, te_l = dset2loader(tr_s), dset2loader(te_s)

        for i, (x, y) in enumerate(tr_l):
            if i != 0:
                break
            print(f'[{name}] Train sample size:')
            print(x.shape, y.shape)

        for i, (x, y) in enumerate(te_l):
            if i != 0:
                break
            print(f'[{name}] Test sample size:')
            print(x.shape, y.shape)

        print("==================== ..Ok")

def test_qskew():
    for name in names:
        print(f"==================== Testing: quantity skewing [{name}]")
        client2dset, te_s = quantity_skew(name, 10)

        tr_s = client2dset[0]
        tr_l, te_l = dset2loader(tr_s), dset2loader(te_s)

        for i, (x, y) in enumerate(tr_l):
            if i != 0:
                break
            print(f'[{name}] Train sample size:')
            print(x.shape, y.shape)

        for i, (x, y) in enumerate(te_l):
            if i != 0:
                break
            print(f'[{name}] Test sample size:')
            print(x.shape, y.shape)
        print("==================== ..Ok")

def test_fskew():
    for name in names:
        print(f"==================== Testing: feature skewing [{name}]")
        # feature_skew_noise(name, 10, 1.)
        client2dset, te_s = feature_skew_filter(name, 5, 3)

        tr_s = client2dset[0]
        tr_l, te_l = dset2loader(tr_s), dset2loader(te_s)

        for i, (x, y) in enumerate(tr_l):
            if i != 0:
                break
            print(f'[{name}] Train sample size:')
            print(x.shape, y.shape)

        for i, (x, y) in enumerate(te_l):
            if i != 0:
                break
            print(f'[{name}] Test sample size:')
            print(x.shape, y.shape)
        print("==================== ..Ok")

if __name__ == "__main__":
    setseed(400)
    test_load()
    test_fskew()
    test_qskew()