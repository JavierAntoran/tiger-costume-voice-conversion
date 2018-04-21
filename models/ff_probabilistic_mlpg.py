import torch
import torch.nn as nn
import numpy as np

class Nloss_GD(nn.Module):
    def __init__(self, dim):
        super(Nloss_GD, self).__init__()
        self.dim = dim

        torch.manual_seed(0)

    def get_likelihoods(self, X, Y, Beta, eps=1e-6):
        # Returns likelihoods of each datapoint for every cluster
        # batch_size
        inv_det = Beta.prod(dim=1)

        if (inv_det < eps).any():
            inv_det += (inv_det < eps).type(torch.cuda.FloatTensor) * eps

        det = 1 / inv_det
        print(torch.sqrt((2 * np.pi) ** self.dim * torch.abs(det)))
        # batch_size
        norm_term = 1 / torch.sqrt((2 * np.pi) ** self.dim * torch.abs(det))
        # batch_size, dims
        inv_covars = Beta
        # batch_size, dims
        dist = (Y - X).pow(2)
        # batch_size
        exponent = (-0.5 * dist * inv_covars).sum(dim=1)
        # batch_size
        pk = norm_term * exponent.exp()
        return pk

    def get_log_likelihoods(self, X, Y, sq_Beta, eps=1e-6):
        # Returns likelihoods of each datapoint for every cluster
        Beta = sq_Beta ** 2
        # batch_size
        log_det_term = 0.5 * (Beta.log().sum(dim=1))
        # print('detterm shape:', log_det_term.shape)
        # 1
        norm_term = -0.5 * np.log(2 * np.pi) * self.dim
        # print('normterm shape:', norm_term.shape)
        # batch_size, dims
        inv_covars = Beta
        # batch_size, dims
        dist = (Y - X).pow(2)
        # batch_size
        exponent = (-0.5 * dist * inv_covars).sum(dim=1)
        # print('exponent shape:', exponent.shape)
        # batch_size
        log_p = (log_det_term + exponent) + norm_term
        # print('log_p shape:', log_p.shape)
        return log_p

    def forward(self, x, y, Beta):
        # Returns -loglike of all data
        # batch_size
        # print(Beta.mean())
        p = self.get_log_likelihoods(x, y, Beta)
        # 1
        E = torch.sum(-p) / x.shape[0]
        return E


class ff_mlpg(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(ff_mlpg, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, int(1.5*n_hid))
        self.fc3 = nn.Linear(int(1.5*n_hid), 2*n_hid)
        self.fc4 = nn.Linear(2*n_hid, output_dim)
        self.drop = nn.Dropout(p=0, inplace=False)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.bn2 = nn.BatchNorm1d(int(1.5*n_hid))
        self.bn3 = nn.BatchNorm1d(2*n_hid)
        

        # choose your non linearity
        #self.act = nn.Tanh()
        #self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        #self.act = nn.ELU(inplace=True)
        #self.act = nn.SELU(inplace=True)

    def forward(self, x):
        # -----------------
        x = self.fc1(x)
        # -----------------
        x = self.act(x)
        # -----------------
        x = self.bn1(x)
        x = self.drop(x)
        # -----------------
        x = self.fc2(x)
        # -----------------
        x = self.act(x)
        # -----------------
        x = self.bn2(x)
        x = self.drop(x)
        # -----------------
        x = self.fc3(x)
        # -----------------
        x = self.bn3(x)
        x = self.drop(x)
        # -----------------
        x = self.act(x)
        # -----------------
        y = self.fc4(x)
        
        out = y[:, :int(self.output_dim/2)]
        sq_Beta = y[:, int(self.output_dim/2):]
        
        return out, sq_Beta
