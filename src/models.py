import torch
from torch import nn
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
