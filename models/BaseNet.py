from __future__ import division, print_function
import numpy as np
import torch
import torch .nn as nn
from src.regression_utils import *

from autoencoder import Autoencoder
from feed_forward import feed_forward
from ff_probabilistic_mlpg import Nloss_GD, ff_mlpg

class BaseNet(object):
    def __init__(self):
        self.epoch = 0

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)

        if not torch.cuda.is_available():
            state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        else:
            state_dict = torch.load(filename)

        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch


class Net(BaseNet):
    def __init__(self, name, input_dim, output_dim, n_hid, n_bottleneck=None, lr=1e-4, weight_decay=0, cuda=True):
        super(Net, self).__init__()
        cprint('c', '\nNet:')
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_bottleneck = n_bottleneck
        self.n_hid = n_hid
        self.cuda = cuda
        self.create_net()
        self.create_opt(lr, weight_decay)

    def create_net(self):

        if self.name == 'feed_forward':
            self.model = feed_forward(self.input_dim, self.output_dim, self.n_hid)
            self.J = nn.MSELoss(size_average=True, reduce=True)
        elif self.name == 'autoencoder':
            self.model = Autoencoder(self.input_dim, self.output_dim, self.n_hid, self.n_bottleneck)
            self.J = nn.MSELoss(size_average=True, reduce=True)
        elif self.name == 'ff_mlpg':
            self.model = ff_mlpg(self.input_dim, self.output_dim, self.n_hid)
            self.J = Nloss_GD(self.input_dim)
        else:
            pass

        if self.cuda:
            self.model = self.model.cuda()
            self.J = self.J.cuda()
        print('    Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

    def create_opt(self, lr=1e-4, weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.schedule = None  # [-1] #[50,200,400,600]

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), volatile=False, cuda=self.cuda)

        self.optimizer.zero_grad()
        if self.name == 'ff_mlpg':
            out, sq_Beta = self.model(x)
            loss = self.J(out, y, sq_Beta)
            loss.backward()
            self.optimizer.step()
            return loss.data[0], sq_Beta.abs().mean().data
        elif self.name == 'feed_forward':
            out = self.model(x)
            loss = self.J(out, y)
        else:
            out = self.model(x)
            loss = self.J(out, y)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y), volatile=True, cuda=self.cuda)

        if self.name == 'ff_mlpg':
            out, sq_Beta = self.model(x)
            loss = self.J(out, y, sq_Beta)
            return loss.data[0], sq_Beta.abs().mean().data
        elif self.name == 'feed_forward':
            out = self.model(x)
            loss = self.J(out, y)
        else:
            out = self.model(x)
            loss = self.J(out, y)
        return loss.data[0]

    def predict(self, x, train=False):
        x, = to_variable(var=(x,), volatile=True, cuda=self.cuda)

        if self.name == 'ff_mlpg':
            out, sq_Beta = self.model(x)
            return out.data, sq_Beta.abs().data
        else:
            out = self.model(x)
            return out.data