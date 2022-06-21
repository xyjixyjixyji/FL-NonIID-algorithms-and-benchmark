from .pydatasets.CIFAR10 import CIFAR10_Dataset
from .pydatasets.KMNIST import KMNIST_Dataset
from .pydatasets.MNIST import MNIST_Dataset
from .pydatasets.SVHN import SVHN_Dataset
import torchvision.transforms as transforms

def preprocess(dataset_name,
               indices=None,
               noise=False,
               noise_mean=0.,
               noise_std=1,
               filter=False,
               filter_sz=3):

    augment_dataset_name = ['cifar10']

    name2func = { #'celeba': CELEBA_Dataset,
                 'cifar10': CIFAR10_Dataset,
                 'kmnist': KMNIST_Dataset,
                 'mnist': MNIST_Dataset,
                 'svhn': SVHN_Dataset}
    
    rootp = './datafiles/datasets/'
    rootp += dataset_name

    dataset_func = name2func[dataset_name]

    if dataset_func == None:
        raise ValueError("DATASET NOT IMPLEMENTED")

    tf_train = [ transforms.ToPILImage(),
                transforms.Resize([32,32]),
                transforms.Grayscale(num_output_channels=3)]

    tf_test = [ transforms.ToPILImage(),
                transforms.Resize([32,32]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    # if dataset_name in augment_dataset_name:
    #     tf_train.append(transforms.RandomCrop(28))
    #     tf_train.append(transforms.RandomHorizontalFlip())

    tf_train.append(transforms.ToTensor())
    tf_train.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    tf_train = transforms.Compose(tf_train)
    tf_test = transforms.Compose(tf_test)

    train_set = dataset_func(rootp=rootp,
                             train=True,
                             transform=tf_train,
                             download=True,
                             indices=indices,
                             noise=noise,
                             noise_mean=noise_mean,
                             noise_std=noise_std,
                             filter=filter,
                             filter_sz=filter_sz)
    
    test_set = dataset_func(rootp=rootp,
                            train=False,
                            transform=tf_train,
                            download=True,
                            indices=indices,
                            noise=noise,
                            noise_mean=noise_mean,
                            noise_std=noise_std,
                            filter=filter,
                            filter_sz=filter_sz)

    
    return train_set, test_set
