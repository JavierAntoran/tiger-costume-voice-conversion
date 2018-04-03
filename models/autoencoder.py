import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, n_bottleneck):
        super(Autoencoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_bottleneck)
        self.fc3 = nn.Linear(n_bottleneck, n_hid)
        self.fc4 = nn.Linear(n_hid, output_dim)

        # choose your non linearity
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x):
        print(type(x.data))
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
        # -----------------

        return y

    def get_features(self, x):
        return None