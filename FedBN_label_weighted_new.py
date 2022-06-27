


"""
Federated learning with different aggregation strategy on benchmark exp.

example test command:
    python FL_tr.py --mode fedbn \
                    --model DigitModel \
                    --dataset mnist \
                    --skew feat_noise \
                    --noise_std 0.5 \
                    --nclient 3

parameters you HAVE TO set:

        mode: fedbn fedprox fed avg
        model: check args.model
        dataset: mnist, kmnist, svhn, cifar10 (case sensitive)
        skew:
            quantity:
                - Di_alpha: the parameter of dirichlet distribution
            feat_noise
                - noise_std: the standard deviation of the noise
            feat_filter
                - filter_sz: the kernel size of the mean filter
            label_across
                - Di_alpha: same as above
                - overlap: if the clients' label are allowed to overlap
            label_within
                - Di_alpha: same as above
        nlabel: please set correspondant to your dataset, for all dataset we offer, default is 10
        nclient: five is generally feasible, more clients will bring slowing speed, since this is an emulation

details in args
"""

#
# COURTESY: PART OF THE CODE ARE REFERENCED FROM THE FOLLOWING LINK
# https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py
#

import torch
import time
import os
import copy
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from models.digit import DigitModel
from models.resnet import *
from skew import label_skew_across_labels, label_skew_by_within_labels, quantity_skew, feature_skew_noise, feature_skew_filter
from datafiles.loaders import dset2loader
from datafiles.utils import setseed
from datafiles.preprocess import preprocess
from tr_utils import train, train_fedprox,train_LW

os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='test the pretrained model')
parser.add_argument('--percent', type=float, default=0.1, help ='percentage of dataset to train')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help ='batch size')
parser.add_argument('--iters', type=int, default=50, help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=3, help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn')
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--load_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--log_path', type=str, default='./logs_label_weighted/', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
parser.add_argument('--choke', action = 'store_true', help='choke those bad clients when communicating')
parser.add_argument('--label', action='store_true', help = 'reweight according to label number in FedBN')
parser.add_argument('--model', type=str, default="DigitModel", help = 'model used:| DigitModel | resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | resnet1202 |')
parser.add_argument('--dataset', type=str, default="mnist", help = '| mnist | kmnist | svhn | cifar10 |')
parser.add_argument('--skew', type=str, default='none', help='| none | quantity | feat_filter | feat_noise | label_across | label_within |')
parser.add_argument('--noise_std', type=float, default=0.5, help='noise level for gaussion noise')
parser.add_argument('--filter_sz', type=int, default=3, help='filter size for filter')
parser.add_argument('--Di_alpha', type=float, default=0.5, help='alpha level for dirichlet distribution')
parser.add_argument('--overlap', type=bool, default=True, help='If lskew_across allows label distribution to overlap')
parser.add_argument('--nlabel', type=int, default=10, help='number of label for dirichlet label skew')
parser.add_argument('--nclient', type=int, default=4, help='client number')
parser.add_argument('--seed', type=int, default=400, help='random seed')
args = parser.parse_args()

print(f"args: {args}")

assert(args.dataset in ['svhn', 'cifar10', 'mnist', 'kmnist'])
assert(args.skew in ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'])
# assert(args.Fl_size % 2 == 1)
assert(args.mode in ['fedavg', 'fedprox', 'fedbn'])

setseed(args.seed)


def prepare_data(args):
    train_loaders = []
    test_loaders  = []
    tr_sets, te_set = [],[]
        
    if args.skew == 'none':
        tr_sets, te_set = feature_skew_noise(args.dataset, args.nclient, 0)
    elif args.skew == 'quantity':
        tr_sets, te_set = quantity_skew(args.dataset, args.nclient, args.Di_alpha)
    elif args.skew == 'feat_noise':
        tr_sets, te_set = feature_skew_noise(args.dataset, args.nclient, args.noise_std)
    elif args.skew == 'feat_filter':
        tr_sets, te_set = feature_skew_filter(args.dataset, args.nclient, args.filter_sz)
    elif args.skew == 'label_across':
        tr_sets, te_set = label_skew_across_labels(args.dataset, args.nclient, args.nlabel, args.Di_alpha, args.overlap)
    elif args.skew == 'label_within':
        tr_sets, te_set = label_skew_by_within_labels(args.dataset, args.nclient, args.nlabel, args.Di_alpha)
    else:
        raise ValueError("UNDEFINED SKEW")

    for tr_s in tr_sets:
        tr_l = dset2loader(tr_s,args.batch_size)
        te_l = dset2loader(te_set,args.batch_size)
        train_loaders.append(tr_l)
        test_loaders.append(te_l)
    # test_loader = dset2loader(te_set,args.batch_size)

    return train_loaders, test_loaders

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, server_model, models, client_weights, train_losses):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            if args.choke and len(train_losses)!=0:
                loss_mean = np.mean(train_losses)
                loss_std = np.std(train_losses, ddof=1)
                if loss_std > 0.2:
                    tmp_total = 0
                    for client_idx in range(len(client_weights)):
                        if(train_losses[client_idx]>(loss_mean+loss_std)):
                            client_weights[client_idx] = 0
                        else:
                            tmp_total += client_weights[client_idx]
                    client_weights = [client_weights[client_idx]/tmp_total for client_idx in range(len(client_weights))]
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    print('Device:', device)

    args.save_path = os.path.join(args.save_path, args.model)
    log_path = os.path.join(args.log_path, args.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}_{}_{}_{}.log'.format(args.mode + "Choking" if args.choke else "" ,args.dataset,args.skew,args.nclient)), 'a')
    if args.label:
        logfile = open(os.path.join(log_path,'{}_{}_{}_{}.log'.format(args.mode + "Labeling",args.dataset,args.skew,args.nclient)), 'w')
    logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('===Setting===\n')
    logfile.write('    lr: {}\n'.format(args.lr))
    logfile.write('    batch: {}\n'.format(args.batch_size))
    logfile.write('    iters: {}\n'.format(args.iters))
    logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}_{}_{}.bin'.format(args.mode,args.dataset,args.skew))
   
   
    server_model = eval(args.model)().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)
    # federated setting
    client_num = args.nclient
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    # if args.test:
    #     print('Loading snapshots...')
    #     checkpoint = torch.load(os.path.join(args.load_path, '{}'.format(args.mode)))
    #     server_model.load_state_dict(checkpoint['server_model'])
    #     if args.mode.lower()=='fedbn':
    #         for client_idx in range(client_num):
    #             models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
    #         for test_idx, test_loader in enumerate(test_loaders):
    #             _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
    #             print(' client {}| Test  Acc: {:.4f}'.format(test_idx, test_acc))
    #     else:
    #         for test_idx, test_loader in enumerate(test_loaders):
    #             _, test_acc = test(server_model, test_loader, loss_fun, device)
    #             print(' client {}| Test  Acc: {:.4f}'.format(test_idx, test_acc))
    #     exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0
    # start training
    train_losses = []
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        samples = [0 for i in range(client_num)]
        total = 0
        labels = torch.tensor([])
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, server_model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    if args.mode.lower() == 'fedavg':
                        samples[client_idx] += len(train_loader)
                        total += len(train_loader)
                    _, _, labels_num = train_LW(model, train_loader, optimizer, loss_fun, client_num, device,args)
                    if wi == 0 :
                        labels = torch.cat((labels,labels_num.unsqueeze(0)),0)

         
        # aggregation
        client_weights = [1/client_num for i in range(client_num)]
        if args.mode.lower() == 'fedavg':
            client_weights = [samples[i]/total for i in range(client_num)]
        if args.label:
            total = 0
            total_label = torch.sum(labels,dim=0)
            client_w = [0 for i in range(client_num)]
            for i in range(args.nlabel):
                for j in range(client_num):
                    client_w[j] += labels[j][i]/total_label[i]
            client_weights = [client_w[i]/args.nlabel for i in range(client_num)]
            print(client_weights)
        server_model, models = communication(args, server_model, models, client_weights, train_losses)
        min_test_loss = 1000
        max_test_acc = 0
        # report after aggregation
        train_losses = []
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model, train_loader, loss_fun, device) 
                train_losses.append(train_loss)
                print(' client {}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(client_idx, train_loss, train_acc))
                logfile.write(' client {}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(client_idx ,train_loss, train_acc))\

        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' client {}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(test_idx, test_loss, test_acc))
            logfile.write(' client {}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(test_idx, test_loss, test_acc))
            if test_acc > max_test_acc:
                server_model = models[test_idx]
                max_test_acc = test_acc
                min_test_loss = test_loss
        print(' server | Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(min_test_loss, max_test_acc))
        logfile.write(' server | Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(min_test_loss, max_test_acc))
        logfile.flush()

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        dic = {'model_{}'.format(num):models[num].state_dict() for num in range(client_num)}
        dic.update({'server_model': server_model.state_dict()})
        torch.save(dic, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)
    logfile.flush()
    logfile.close()
