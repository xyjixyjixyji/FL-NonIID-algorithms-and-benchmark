import torchvision.datasets as datasets
from PIL import Image
from .datasets import GeneralDataset
from ..utils import add_gaussian_noise

class MNIST_Dataset(GeneralDataset):
    
    def __init__(self,
                 rootp,
                 train,
                 transform=None, 
                 target_transform=None,
                 download=False,
                 indices=None,
                 noise=False,
                 noise_mean=0.,
                 noise_std=1.):

        self.root = rootp # dataset rootpath
        self.train = train # train?
        self.tf = transform # tf(x)
        self.ttf = target_transform # ttf(y)
        self.dld = download # True when you run the first time
        self.indices = indices # which part of dset you want?

        self.noise = noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.x, self.y = self.download_dataset(self.root,
                                               self.train,
                                               self.tf,
                                               self.ttf,
                                               self.dld)
    
    def download_dataset(self,
                         root,
                         train,
                         tf,
                         ttf,
                         dld):
        # download dataset to root
        obj = datasets.MNIST(root, train, tf, ttf, dld)

        x, y = obj.data, obj.targets

        if self.indices is not None and self.train:
            x = obj.data[self.indices]
            y = obj.targets[self.indices]
        
        return x, y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]

        if self.tf:
            x = self.tf(x)
        if self.ttf:
            y = self.ttf(y)

        if self.noise:
            x = add_gaussian_noise(x,
                                   mean=self.noise_mean,
                                   std=self.noise_std)
        
        return x, y