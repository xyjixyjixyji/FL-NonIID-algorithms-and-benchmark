from torch.utils.data import Dataset

class GeneralDataset(Dataset):

    def __init__(self, **kwargs):
        pass
    
    def download_dataset(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def set_indices(self, indices):
        if not self.indices:
            self.indices = indices
            self.x = self.x[indices]
            self.y = self.y[indices]