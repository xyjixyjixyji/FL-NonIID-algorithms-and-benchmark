


"""
Federated learning with different aggregation strategy on benchmark exp.

example test command:
    python FL_tr.py --mode fedbn --model DigitModel --dsetname mnist --skew feat_noise --noise_std 0.5 --nclient 3

parameters you HAVE TO set:

        mode: fedbn fedprox fed avg
        model: check args.model
        dsetname: mnist, kmnist, svhn, cifar10 (case sensitive)
        skew:
            quantity:
                - Di_alpha: the parameter of dirichlet distribution
            feat_noise
                - noise_std: the standard deviation of the noise
            feat_filter
                - filter_sz: the kernel size of the mean filter
            label_across
                - Di_alpha: same as above
            label_within
                - Di_alpha: same as above
        nlabel: please set correspondant to your dataset, for all dataset we offer, default is 10
        nclient: five is generally feasible, more clients will bring slowing speed, since this is an emulation

details in args
"""

import torch
import time
import os
import copy
import torch.nn
import torch.optim
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

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='test the pretrained model')
parser.add_argument('--percent', type=float, default=0.1, help ='percentage of dataset to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help ='batch size')
parser.add_argument('--iters', type=int, default=100, help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=1, help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn')
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type=str, default='../checkpoint', help='path to save the checkpoint')
parser.add_argument('--load_path', type=str, default='../checkpoint', help='path to save the checkpoint')
parser.add_argument('--log_path', type=str, default='../logs/', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

parser.add_argument('--model', type = str, default="DigitModel", help = 'model used:| DigitModel | resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | resnet1202 |')
parser.add_argument('--dsetname', type = str, default="mnist", help = '| mnist | kmnist | svhn | cifar10 |')
parser.add_argument('--skew', type=str, default='None', help='| none | quantity | feat_filter | feat_noise | label_across | label_within |')
parser.add_argument('--noise_std', type=float, default=0.5, help='noise level for gaussion noise')
parser.add_argument('--filter_sz', type=int, default=3, help='filter size for filter')
parser.add_argument('--Di_alpha', type=float, default=0.5, help='alpha level for dirichlet distribution')
parser.add_argument('--nlabel', type = int, default=10, help = 'number of label for dirichlet label skew')
parser.add_argument('--nclient', type = int, default=5, help = 'client number')

args = parser.parse_args()

print(f"args: {args}")

assert(args.dsetname in ['svhn', 'cifar10', 'mnist', 'kmnist'])
assert(args.skew in ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'])
assert(args.Fl_size % 2 == 1)
assert(args.mode in ['fedavg', 'fedprox', 'fedbn'])

def prepare_data(args):
    train_loaders = []
    test_loaders  = []
    tr_sets, te_set = [],[]
        
    if args.skew == 'none':
        tr_sets, te_set = feature_skew_noise(args.dsetname, args.nclient, 0)
    elif args.skew == 'quantity':
        tr_sets, te_set = quantity_skew(args.dsetname, args.nclient, args.Di_alpha)
    elif args.skew == 'feat_noise':
        tr_sets, te_set = feature_skew_noise(args.dsetname, args.nclient, args.noise_std)
    elif args.skew == 'feat_filter':
        tr_sets, te_set = feature_skew_filter(args.dsetname, args.nclient, args.filter_sz)
    elif args.skew == 'label_across':
        tr_sets, te_set = label_skew_across_labels(args.dsetname, args.nclient, args.nlabel, args.Di_alpha)
    elif args.skew == 'within_label':
        tr_sets, te_set = label_skew_by_within_labels(args.dsetname, args.nclient, args.nlabel, args.Di_alpha)
    else:
        raise ValueError("UNDEFINED SKEW")

    for tr_s in tr_sets:
        tr_l = dset2loader(tr_s,args.batch_size)
        te_l = dset2loader(te_set,args.batch_size)
        train_loaders.append(tr_l)
        test_loaders.append(te_l)

    return train_loaders, test_loaders

def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
       
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)

    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)

        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

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
def communication(args, server_model, models, client_weights):
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
    logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
    logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logfile.write('===Setting===\n')
    logfile.write('    lr: {}\n'.format(args.lr))
    logfile.write('    batch: {}\n'.format(args.batch_size))
    logfile.write('    iters: {}\n'.format(args.iters))
    logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
   
   
    server_model = eval(args.model)().to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)
    
    # federated setting
    client_num = args.nclient
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load(os.path.join(args.load_path, '{}'.format(args.mode)))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' client {}| Test  Acc: {:.4f}'.format(test_idx, test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' client {}| Test  Acc: {:.4f}'.format(test_idx, test_acc))
        exit(0)

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
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device)
                else:
                    train(model, train_loader, optimizer, loss_fun, client_num, device)
         
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)
        
        # report after aggregation
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model, train_loader, loss_fun, device) 
                print(' client {}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(client_idx ,train_loss, train_acc))
                logfile.write(' client {}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(client_idx ,train_loss, train_acc))\

        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' client {}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(test_idx, test_loss, test_acc))
            logfile.write(' client {}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(test_idx, test_loss, test_acc))

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        dic = {'model_{}'.fomat(num):models[num].state_dict() for num in range(client_num)}
        dic.update({'server_model': server_model.state_dict()})
        torch.save(dic, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)

    logfile.flush()
    logfile.close()


