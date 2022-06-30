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
from models.digit import DigitModel, MoonDigitModel
from models.resnet import *
from skew import label_skew_across_labels, label_skew_by_within_labels, quantity_skew, feature_skew_noise, feature_skew_filter, prepare_data
from datafiles.loaders import dset2loader
from datafiles.utils import setseed
from datafiles.preprocess import preprocess
from tr_utils import train, train_fedprox


# for GPU server selection
os.environ['CUDA_VISIBLE_DEVICES']='1'

#
# COURTESY:
# https://github.com/QinbinLi/MOON
#

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='test the pretrained model')
parser.add_argument('--percent', type=float, default=0.001, help ='percentage of dataset to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help ='batch size')
parser.add_argument('--iters', type=int, default=50, help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=3, help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='moon', help='fedavg | fedprox | fedbn | moon')
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--load_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--log_path', type=str, default='./log_moon/', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true',default=False, help='resume training from the save path checkpoint')
parser.add_argument('--model', type=str, default="MoonDigitModel", help = 'model used:| MoonDigitModel | resnet20 | resnet32 | resnet44 | resnet56 | resnet110 | resnet1202 |')
parser.add_argument('--dataset', type=str, default="mnist", help = '| mnist | kmnist | svhn | cifar10 |')
parser.add_argument('--skew', type=str, default="quantity", help='| none | quantity | feat_filter | feat_noise | label_across | label_within |')
parser.add_argument('--noise_std', type=float, default=0.5, help='noise level for gaussion noise')
parser.add_argument('--filter_sz', type=int, default=3, help='filter size for filter')
parser.add_argument('--Di_alpha', type=float, default=0.5, help='alpha level for dirichlet distribution')
parser.add_argument('--overlap', type=bool, default=True, help='If lskew_across allows label distribution to overlap')
parser.add_argument('--nlabel', type=int, default=10, help='number of label for dirichlet label skew')
parser.add_argument('--nclient', type=int, default=4, help='client number')
parser.add_argument('--seed', type=int, default=400, help='random seed')
parser.add_argument('--cuda', type=bool, default=True, help='if cuda is available' )
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
parser.add_argument('--glo_lr', type=float, default=0.001, help='global learning rate')
parser.add_argument('--reg_lamb', type=float, default=1.0, help='the moon parameter')
args = parser.parse_args()

assert(args.dataset in ['svhn', 'cifar10', 'mnist', 'kmnist'])
assert(args.skew in ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'])
assert(args.mode in ['fedavg', 'fedprox', 'fedbn', 'moon'])
setseed(args.seed)
log_path = os.path.join(args.log_path, args.model)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = open(os.path.join(log_path,'{}_{}_{}.log'.format(args.mode,args.dataset,args.skew)), 'w')

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class MOON():
    def __init__(
        self,
        model,
        args
    ):
        self.model = model
        self.args = args

        self.clients = args.nclient

        # copy private models for each client
        self.client_models = {}
        for client in range(self.clients):
            self.client_models[client] = copy.deepcopy(
                model.cpu()
            )

        # to cuda
        if self.args.cuda is True:
            self.model = self.model.cuda()

        # construct dataloaders
        self.train_loaders, self.test_loaders = prepare_data(args)


    def train(self):
        # Training
        max_acc = 0
        min_loss = 10000
        for r in range(1, self.args.iters + 1):
            print("============ Train epoch {} ============".format(r))
            logfile.write("============ Train epoch {} ============\n".format(r))
            local_models = {}

            avg_loss = Averager()
            # all_per_accs = []
            for client in range(self.clients):
                # to cuda
                if self.args.cuda is True:
                    self.client_models[client].cuda()

                local_model, per_acc, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    local_model=copy.deepcopy(self.client_models[client]),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )
                print(' client {}| Loss: {:.4f} | Test  Acc: {:.4f}'.format(client, loss, per_acc))
                logfile.write(
                    ' client {}| Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(client, loss, per_acc))
                local_models[client] = copy.deepcopy(local_model)

                # update local model
                self.client_models[client] = copy.deepcopy(local_model)

                avg_loss.add(loss)
                if per_acc > max_acc:
                    max_acc = per_acc
                    min_loss = loss

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            print(' server  | Loss: {:.4f} | Test  Acc: {:.4f}'.format( min_loss, max_acc))
            logfile.write(' server  | Loss: {:.4f} | Test  Acc: {:.4f}'.format( min_loss, max_acc))


    def update_local(self, r, model, local_model, train_loader, test_loader):
        glo_model = copy.deepcopy(model)
        glo_model.eval()
        local_model.eval()

        optimizer = torch.optim.SGD(params=model.parameters(), lr=self.args.lr)

        n_total_bs = self.args.wk_iters*len(train_loader)

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            

            model.train()
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            #print(batch_x.shape)
            hs, logits = model(batch_x)
            hs1, _ = glo_model(batch_x)
            hs0, _ = local_model(batch_x)

            criterion = nn.CrossEntropyLoss()
            ce_loss = criterion(logits, batch_y.long())

            # moon loss
            ct_loss = self.contrastive_loss(
                hs, hs0.detach(), hs1.detach()
            )

            loss = ce_loss + self.args.reg_lamb * ct_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        per_acc = self.test(
                model=model,
                loader=test_loader,
            )
        loss = avg_loss.item()
        return model, per_acc, loss

    def contrastive_loss(self, hs, hs0, hs1):
        cs = nn.CosineSimilarity(dim=-1)
        sims0 = cs(hs, hs0)
        sims1 = cs(hs, hs1)

        sims = 2.0 * torch.stack([sims0, sims1], dim=1)
        labels = torch.LongTensor([1] * hs.shape[0])
        labels = labels.to(hs.device)

        criterion = nn.CrossEntropyLoss()
        ct_loss = criterion(sims, labels)
        return ct_loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

        def save_checkpoints(self, fpath):
            torch.save({
                'server_model': self.model.state_dict(),
            }, fpath)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', device)
    args.save_path = os.path.join(args.save_path, args.model)
    log_path = os.path.join(args.log_path, args.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'{}_{}_{}_{}.log'.format(args.mode ,args.dataset,args.skew,args.nclient)), 'w')
    
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
    
    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
    moon = MOON(server_model, args)
    moon.train()
    moon.save_checkpoints(SAVE_PATH)
    logfile.flush()
    logfile.close()

