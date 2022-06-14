'''
skewing the datasets
'''
from datafiles.preprocess import preprocess
import numpy.random as random
import numpy as np

def feature_skew(dataset_name,
                 noise_std,):
    '''
        Feature skew, adding random gaussian noise to the input
        skewing level controlled by noise standard deviation
    '''
    tr_set, te_set = preprocess(dataset_name,
                                noise=True,
                                noise_std=noise_std)
    return tr_set, te_set

def quantity_skew(dataset_name, nclient, alpha=0.5):
    '''
    Dirichlet distribution, to nclient
    '''
    client2dataset = {}

    tr_set, te_set = preprocess(dataset_name)
    nsample = tr_set.y.shape[0]

    indices = random.permutation(nsample)

    # normally the batchsize is 32, we want every client to have at least 32 samples
    min = float('-inf')
    while min < 32:
        prop = random.dirichlet([alpha] * nclient)
        prop = prop / prop.sum()
        min = np.min(prop * len(indices))
    prop = (np.cumsum(prop) * len(indices)).astype(int)[:-1]

    indices = np.split(indices, prop)

    for i in range(nclient):
        tr_set, _ = preprocess(dataset_name=dataset_name,
                               indices=indices[i],
                               noise=False)
        client2dataset[i] = tr_set
    
    for i in range(nclient):
        print(f'Client{i} has {len(client2dataset[i])} samples')
    
    print(f"Done skewing [{dataset_name}]")
    return client2dataset, te_set
    
def label_skew(dataset_name, nclient, y_tr, alpha):
    pass