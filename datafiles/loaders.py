'''
From datasets to dataloaders
'''


from torch.utils.data import DataLoader

def dset2loader(dataset, batch_size=32):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)
    return loader
