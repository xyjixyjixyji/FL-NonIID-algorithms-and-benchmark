'''
From datasets to dataloaders
'''

from .pydatasets.CELEBA import CELEBA_Dataset
from .pydatasets.CIFAR10 import CIFAR10_Dataset
from .pydatasets.KMNIST import KMNIST_Dataset
from .pydatasets.MNIST import MNIST_Dataset
from .pydatasets.SVHN import SVHN_Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

def dset2loader(dataset_name,
                indices=None,
                batch_size=64,
                noise=False,
                noise_mean=0.,
                noise_std=1.):

    augment_dataset_name = ['cifar10'] # some datasets need augmentation
    PIL_dataset_name = ['kmnist'] # some datasets load in with PIL image not tensor

    name2func = {'celeba': CELEBA_Dataset,
                 'cifar10': CIFAR10_Dataset,
                 'kmnist': KMNIST_Dataset,
                 'mnist': MNIST_Dataset,
                 'svhn': SVHN_Dataset}
    
    rootp = './data/datasets/'
    name2rootp = {
        'celeba': rootp + 'celaba',
        'cifar10': rootp + 'cifar10',
        'kmnist': rootp + 'kmnist',
        'mnist': rootp + 'mnist',
        'svhn': rootp + 'svhn',
    }
    
    rootp = name2rootp[dataset_name]
    dataset_func = name2func[dataset_name]

    if dataset_func == None:
        raise ValueError("DATASET NOT IMPLEMENTED")

    tf_train = []
    tf_test = []

    if dataset_name in PIL_dataset_name:
        tf_train.append(transforms.ToTensor())
        tf_test.append(transforms.ToTensor())
        

    if dataset_name in augment_dataset_name:
        tf_train.append(transforms.ToPILImage())
        tf_train.append(transforms.RandomCrop(32))
        tf_train.append(transforms.RandomHorizontalFlip())
        tf_train.append(transforms.ToTensor())
    
    tf_train = transforms.Compose(tf_train)
    tf_test = transforms.Compose(tf_test)

    train_set = dataset_func(rootp=rootp,
                             train=True,
                             transform=tf_train,
                             download=True,
                             indices=indices,
                             noise=noise,
                             noise_mean=noise_mean,
                             noise_std=noise_std)

    test_set = dataset_func(rootp=rootp,
                            train=False,
                            transform=tf_train,
                            download=True,
                            indices=indices,
                            noise=noise,
                            noise_mean=noise_mean,
                            noise_std=noise_std)
    
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_loader, test_loader
