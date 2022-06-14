from torch.utils.data import Dataset

class GeneralDataset(Dataset):

    def __init__(self, **kwargs):
        self.x, self.y = self.download_dataset(**kwargs)
    
    def download_dataset(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.x)
