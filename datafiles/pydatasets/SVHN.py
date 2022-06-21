import torchvision.datasets as datasets
import torch
import torch.nn.functional as F
from .datasets import GeneralDataset
from ..utils import add_gaussian_noise

class SVHN_Dataset(GeneralDataset):
    
    def __init__(self,
                 rootp,
                 train,
                 transform=None, 
                 target_transform=None,
                 download=False,
                 indices=None,
                 noise=False,
                 noise_mean=0.,
                 noise_std=1.,
                 filter=False,
                 filter_sz=3):

        self.root = rootp # dataset rootpath
        self.train = 'train' if train else 'test' # train?
        self.tf = transform # tf(x)
        self.ttf = target_transform # ttf(y)
        self.dld = download # True when you run the first time
        self.indices = indices # which part of dset you want?

        self.noise = noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.filter = filter
        self.filter_sz = filter_sz

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
        obj = datasets.SVHN(root, train, tf, ttf, dld)

        x, y = obj.data, obj.labels
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.float(), y.float()

        if len(x.shape) == 3: # (B, H, W)
            x = torch.unsqueeze(x, 1)

        if self.indices is not None and self.train == 'train':
            x = x[self.indices]
            y = y[self.indices]
        
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.filter:
            x = x.to(device)
            y = y.to(device)
            sz = [[1 for _ in range(self.filter_sz)] for _ in range(self.filter_sz)]
            filt = torch.tensor(sz) // (self.filter_sz ** 2)
            filt = filt.expand(3, 3, self.filter_sz, self.filter_sz)
            filt.to(device)
            x = x.view(1,3,32,32)
            x = F.conv2d(x, filt, stride=1, padding=1)
            x = x.squeeze(0)
        
        return x, y
