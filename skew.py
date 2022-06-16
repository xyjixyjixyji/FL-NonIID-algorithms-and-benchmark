'''
    Skewing the datasets

        All functions return (client2set, test_set)

    How to use:
        use any of the functions, says feature_skew_noise()

        using client2dataset, te_set = feature_skew_noise('mnist',
                                                          5,
                                                          .5)
        
        we can index the **client2dataset** for corespondding dataset
            - client2dataset[0] = dataset for client 0
            - client2dataset[i] = dataset for client i
        we also have the test_set te_set
        
        then we uses loader(dataset, batch_size) to transform our
        datasets (for each client) to a dataloader

        Further steps are handed to regular Federated Learning Simulation

        Using the dictionary to acquire the dataset is all
'''

from datafiles.preprocess import preprocess
import numpy.random as random
import numpy as np

def feature_skew_noise(dataset_name,
                       nclient,
                       noise_std=.5,):
    '''
        Feature skew, adding random gaussian noise to the input
        skewing level controlled by noise standard deviation (min: 0, max: 1)
    '''
    te_set = None
    client2dataset = {}
    for i in range(nclient):
        tr_s, te_s = preprocess(dataset_name=dataset_name,
                                noise=True,
                                noise_std=noise_std)
        client2dataset[i] = tr_s
        if i == 0:
            te_set = te_s
    return client2dataset, te_set


def feature_skew_filter(dataset_name,
                        nclient,
                        filter_sz=3,):
    '''
        Feature skew, using filters to filter the dataset, 
        skewing level controlled by filter size (min: 1, max: 5)
    '''
    te_set = None
    client2dataset = {}
    for i in range(nclient):
        tr_s, te_s = preprocess(dataset_name=dataset_name,
                                filter=True,
                                filter_sz=filter_sz)
        client2dataset[i] = tr_s
        if i == 0:
            te_set = te_s
    return client2dataset, te_set


def quantity_skew(dataset_name,
                  nclient,
                  alpha=0.5):
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
    
# each client holds some labels, following dirichlet dist.
def label_skew_across_labels(dataset_name, nclient, nlabel, alpha):
    client2dataset = {}

# for each label, each clients hold a certain # of samples, following dirichlet dist.
def label_skew_by_within_labels(dataset_name, nclient, nlabel, alpha):
    client2dataset = {}
