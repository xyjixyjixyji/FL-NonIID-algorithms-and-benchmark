'''
Test Suite and Debugging for ./data/*
'''

from data.loaders import dset2loader

names = ['mnist', 'kmnist', 'svhn', 'cifar10']
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
