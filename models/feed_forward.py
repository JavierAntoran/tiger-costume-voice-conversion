import torch.nn.functional as F
import torch
import torch.nn as nn

class feed_forward(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(feed_forward, self).__init__()

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, output_dim)
        self.log_sig = nn.LogSigmoid()

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x):
        # -----------------
        x = self.fc1(x)
        # -----------------
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        # -----------------
        x = self.act(x)
        # -----------------
        x = self.fc3(x)
        # -----------------
        x = self.act(x)
        # -----------------
        y = self.fc4(x)
        return y
