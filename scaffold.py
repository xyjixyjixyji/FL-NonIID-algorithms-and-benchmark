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
from tr_utils import train, train_fedprox
from FedBN import prepare_data

# https://github.com/ramshi236/Accelerated-Federated-Learning-Over-MAC-in-Heterogeneous-Networks

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='test the pretrained model')
parser.add_argument('--percent', type=float, default=0.1, help ='percentage of dataset to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help ='batch size')
parser.add_argument('--iters', type=int, default=50, help='iterations for communication')
parser.add_argument('--wk_iters', type=int, default=3, help='optimization iters in local worker between communication')
parser.add_argument('--mode', type=str, default='scaffold', help='fedavg | fedprox | fedbn | scaffold')
parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
parser.add_argument('--save_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--load_path', type=str, default='./checkpoint', help='path to save the checkpoint')
parser.add_argument('--log_path', type=str, default='./logs/', help='path to save the checkpoint')
parser.add_argument('--resume', action='store_true',default=False, help='resume training from the save path checkpoint')
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
parser.add_argument('--max_round', type=int, default=100, help='max round')
parser.add_argument('--test_round', type=int, default=10, help='test round')
parser.add_argument('--weight_decay', type=int, default=0, help='test round')
parser.add_argument('--cuda', type=bool, default=True, help='if cuda is available' )
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
parser.add_argument('--glo_lr', type=float, default=0.001, help='global learning rate')
args = parser.parse_args()

# print(f"args: {args}")

assert(args.dataset in ['svhn', 'cifar10', 'mnist', 'kmnist'])
assert(args.skew in ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'])
assert(args.mode in ['fedavg', 'fedprox', 'fedbn', 'scaffold'])
setseed(args.seed)
log_path = os.path.join(args.log_path, args.model)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = open(os.path.join(log_path,'{}_{}_{}.log'.format(args.mode,args.dataset,args.skew)), 'a')

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



class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(
            lr=lr, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        ng = len(self.param_groups[0]["params"])
        names = list(server_control.keys())

        # BatchNorm: running_mean/std, num_batches_tracked
        names = [name for name in names if "running" not in name]
        names = [name for name in names if "num_batch" not in name]

        t = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                c = server_control[names[t]]
                ci = client_control[names[t]]

                # print(names[t], p.shape, c.shape, ci.shape)
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p.data * group["lr"]
                t += 1
        assert t == ng
        return loss


class Scaffold():
    def __init__(
        self, model, args
    ):
        self.model = model
        self.args = args
        self.clients = args.nclient

        # construct dataloaders
        self.train_loaders, self.test_loaders = prepare_data(args)

        # control variates
        self.server_control = self.init_control(model)
        self.set_control_cuda(self.server_control, True)

        self.client_controls = {
            client: self.init_control(model) for client in range(self.clients)
        }

    def set_control_cuda(self, control, cuda=True):
        for name in control.keys():
            if cuda is True:
                control[name] = control[name].cuda()
            else:
                control[name] = control[name].cpu()

    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).cpu() for name, p in model.state_dict().items()
        }
        return control

    def train(self):
        # Training
        max_acc = 0
        min_loss = 10000
        for r in range(1, self.args.max_round + 1):
            print("============ Train epoch {} ============".format(r))
            logfile.write("============ Train epoch {} ============\n".format(r))
            delta_models = {}
            delta_controls = {}

            for client in range(self.clients):
                # control to gpu
                self.set_control_cuda(self.client_controls[client], True)
                # update local with control variates / ScaffoldOptimizer
                delta_model, per_acc, local_steps, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                    server_control=self.server_control,
                    client_control=self.client_controls[client],
                )

                print(' client {}| Loss: {:.4f} | Test  Acc: {:.4f}'.format(client, loss, per_acc))
                logfile.write(
                    ' client {}| Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(client, loss, per_acc))

                client_control, delta_control = self.update_local_control(
                    delta_model=delta_model,
                    server_control=self.server_control,
                    client_control=self.client_controls[client],
                    steps=local_steps,
                    lr=self.args.lr,
                )
                self.client_controls[client] = copy.deepcopy(client_control)

                delta_models[client] = copy.deepcopy(delta_model)
                delta_controls[client] = copy.deepcopy(delta_control)

                # control to cpu
                self.set_control_cuda(self.client_controls[client], False)
                if per_acc > max_acc:
                    max_acc = per_acc
                    min_loss = loss


            self.update_global(
                r=r,
                global_model=self.model,
                delta_models=delta_models,
            )

            new_control = self.update_global_control(
                r=r,
                control=self.server_control,
                delta_controls=delta_controls,
            )

            self.server_control = copy.deepcopy(new_control)

            print(' server  | Loss: {:.4f} | Test  Acc: {:.4f}'.format(min_loss, max_acc))
            logfile.write(' server  | Loss: {:.4f} | Test  Acc: {:.4f}'.format(min_loss, max_acc))



    def get_delta_model(self, model0, model1):
        """ return a dict: {name: params}
        """
        state_dict = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict

    def update_local(
            self, r, model, train_loader, test_loader,
            server_control, client_control):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        glo_model = copy.deepcopy(model)

        optimizer = ScaffoldOptimizer(
            model.parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay
        )

        n_total_bs = self.args.wk_iters*len(train_loader)

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        for t in range(n_total_bs):

            model.train()
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y.long())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )

            optimizer.step(
                server_control=server_control,
                client_control=client_control
            )

            avg_loss.add(loss.item())


        delta_model = self.get_delta_model(glo_model, model)

        loss = avg_loss.item()
        local_steps = n_total_bs
        per_acc = self.test(
            model=model,
            loader=test_loader,
        )

        return delta_model, per_acc, local_steps, loss

    def update_local_control(
            self, delta_model, server_control,
            client_control, steps, lr):
        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in delta_model.keys():
            c = server_control[name]
            ci = client_control[name]
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci
        return new_control, delta_control

    def update_global(self, r, global_model, delta_models):
        state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in delta_models.keys():
                vs.append(delta_models[client][name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
                vs = param - self.args.glo_lr * mean_value
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
                vs = param - self.args.glo_lr * mean_value
                vs = vs.long()

            state_dict[name] = vs

        global_model.load_state_dict(state_dict, strict=True)

    def update_global_control(self, r, control, delta_controls):
        new_control = copy.deepcopy(control)
        for name, c in control.items():
            mean_ci = []
            for _, delta_control in delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc


    def save_checkpoints(self,fpath):
        torch.save({
            'server_model': self.model.state_dict(),
        }, fpath)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', device)
    log_path = os.path.join(args.log_path, args.model)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path,'{}_{}_{}.log'.format(args.mode,args.dataset,args.skew))
    args.save_path = os.path.join(args.save_path, args.model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
    server_model = eval(args.model)().to(device)
    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
    scaffold = Scaffold(server_model, args)
    scaffold.train()
    scaffold.save_checkpoints(SAVE_PATH)
    logfile.flush()
    logfile.close()




