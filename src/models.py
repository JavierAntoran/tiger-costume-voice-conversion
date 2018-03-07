import torch
from torch import nn
from utils import *
#
# Regression. Dense linear model. Fitted with ML.
# p(w|alpha, Beta, D): p(D|w, Beta)*p(W|alpha)
#
# * Adjustable Weight decay -> 0 Mean Normal prior on weights: W ~N(0, Alpha^-1)
#   to avoid fitting noise. Also ensures matrix invertibility.
# * Adjustable Beta uncertainty factor to model noise. t ~N(y=w*x, Beta^-1)
# * Noise sensibility ~Beta/Alpha
#
# Residuals converge to noise variance with enough training points
#
class dense_linear_LS(nn.Module):
    def __init__(self):
        super(dense_linear_LS, self).__init__()
        self.W = None  # Weights

    def forward(self, x):
        # x = inputs (Nsamples, dim)
        # y = outputs (Nsamples, dim)
        bx = torch.ones(x.shape[0], 1).type(torch.DoubleTensor)
        x = torch.cat((bx, x), 1).t()
        y = self.W.t().matmul(x)
        return y

    def fit(self, x, t, alpha=0.0000001, beta=1, bias=True):
        # x = inputs (Nsamples, dim)
        # t = targets (Nsamples, dim)
        # residuals = output (Nsamples, dim)
        bx = torch.ones(x.shape[0], 1).type(torch.DoubleTensor)
        a = torch.cat((bx, x), dim=1)
        wd = alpha * torch.eye(a.shape[1]).type(torch.DoubleTensor)
        inner = torch.matmul(a.t(), a)
        pseusoInvX = torch.inverse(wd + beta * inner).matmul(beta * a.t())
        self.W = pseusoInvX.matmul(t)
        if bias == False:
            self.W[0, :] = 0
        residuals = t - self.forward(x)
        return residuals

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
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch


class one_hidden_layer(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(one_hidden_layer, self).__init__()

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, output_dim)

        # choose your non linearity
        # self.act = nn.Tanh()
        self.act = nn.Sigmoid()
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # -----------------
        a = self.fc1(x)
        # -----------------
        z = self.act(a)
        # -----------------
        y = self.fc2(z)
        return y


class Net(BaseNet):
    def __init__(self, input_dim, output_dim, n_hid, lr=1e-4, cuda=True):
        super(Net, self).__init__()
        cprint('c', '\nNet:')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hid = n_hid
        self.cuda = cuda
        self.create_net()
        self.create_opt(lr)

    def create_net(self):
        self.model = one_hidden_layer(self.input_dim, self.output_dim, self.n_hid)
        self.J = nn.MSELoss(size_average=True, reduce=True)
        if self.cuda:
            self.model.cuda()
            self.J.cuda()
        print('    Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

    def create_opt(self, lr=1e-4):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.schedule = None  # [-1] #[50,200,400,600]

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), volatile=False, cuda=self.cuda)

        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.J(out, y)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y), volatile=True, cuda=self.cuda)

        out = self.model(x)
        loss = self.J(out, y)
        return loss.data[0]

    def predict(self, x, train=False):
        x, = to_variable(var=(x,), volatile=True, cuda=self.cuda)

        out = self.model(x)
        return out.data
