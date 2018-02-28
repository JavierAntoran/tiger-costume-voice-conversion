import torch
from torch import nn
import numpy as np
from src.models import dense_linear_LS

# Fits fully connected linear model with bias

np.random.seed(1337)
Noise_var = 5
dims = 3

Ntrain = 1000

x_t = 5 * np.ones((Ntrain, dims))
t_t = 40 * np.ones((Ntrain, dims)) + np.sqrt(Noise_var) * np.random.randn(Ntrain, dims)

x_train = torch.from_numpy(x_t)
t_train = torch.from_numpy(t_t)

model = dense_linear_LS()

res = model.fit(x_train, t_train, beta=1, bias=False)
residual_var = res.pow(2).mean()
print('residual variance = %f' % residual_var)
print(model.W)

x_test = 5 * torch.ones(1, dims).type(torch.DoubleTensor)
print(model(x_test))
