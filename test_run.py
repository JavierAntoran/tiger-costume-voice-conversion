from __future__ import print_function
from __future__ import division
import time, sys

import torch
import numpy as np
from src.mlpg import *
from src.regression_utils import *
from models.BaseNet import Net
######

def run_mlpg_regression(x_in, statsdir, savedir='model_saves/theta_best_mlpg.dat'):

    mcsize_mlpg = 181
    np.random.seed(1337)  # for reproducibility

    ## ----------------------------------------------------------------------------------------------------------------
    # read data
    cprint('c', '\nData:')

    ds_stats = np.load(statsdir)
    Xmeans = ds_stats[0]
    Xstds = ds_stats[1]
    Tmeans = ds_stats[3]
    Tstds = ds_stats[4]

    print('  x_test: %d vectors of audio of dim: %d' % (x_in.shape[0], x_in.shape[1]))

    x = get_mtx_deltas(x_in, filter1, filter2)

    x, _, _, in_uv = feature_mv_norm(x, [1, mcsize_mlpg + 1], Xmeans, Xstds)

    ## ---------------------------------------------------------------------------------------------------------------------
    # net dims
    input_dim = x.shape[1]
    output_dim = 2 * input_dim
    n_hid = int(5*input_dim)
    lr = 1e-4
    weight_decay = 1e-7
    use_cuda = torch.cuda.is_available()

    # weight_decay = 0
    # lr = 1e-4

    print('  input_dim: %s' % str(input_dim))
    print('  output_dim: %d' % output_dim)

    # --------------------
    net = Net('ff_mlpg', input_dim, output_dim, n_hid, lr=lr, weight_decay=weight_decay, cuda=use_cuda)
    net.load(savedir + 'theta_best_mlpg.dat')

    ## ---------------------------------------------------------------------------------------------------------------------
    # test

    batch_size = 64

    cprint('c', '\nTest:')

    cost_test = 0

    nb_samples_test = len(x)

    result_ = np.empty((0, input_dim))
    sq_Betas = np.empty((0, input_dim))
    # ----
    tic = time.time()
    for ind in generate_ind_batch(nb_samples_test, batch_size, random=False):
        out, sq_Beta = net.predict(x[ind])
        result_ = np.concatenate((result_, out), axis=0)
        sq_Betas = np.concatenate((sq_Betas, sq_Beta), axis=0)

        # TODO un-normalize sq_betas with vars
    toc = time.time()
    # ----
    print('output feature shape:', result_.shape)
    print('sq_Betas feature shape:', sq_Betas.shape)

    cprint('r', ' time: %f seconds\n' % (toc - tic))

    # Un-norm features without setting null f0
    # TODO: dont apply derivatives to F0
    result_un = feature_mv_unnorm(result_, [1, mcsize_mlpg + 1], Tmeans, Tstds, np.zeros(result_.shape[0]))
    sq_Betas_un = feature_mv_unnorm(sq_Betas, [1, mcsize_mlpg + 1], np.zeros(sq_Betas.shape[1]), 1 / Tstds,
                                    np.zeros(sq_Betas.shape[0]))
    Betas_t = 1 / np.tile(Tstds ** 2, (result_un.shape[0], 1))
    Betas_net = sq_Betas_un ** 2

    print(result_.shape)
    W1 = get_delta_mtx(filter1, result_.shape[0])
    W2 = get_delta_mtx(filter2, result_.shape[0])
    print((sq_Betas_un).mean())
    print(result_un.shape)

    result_mlpg = my_mlpg(result_un, 61, W1, W2, Betas_t)
    result_mlpg[np.squeeze(in_uv) == 1, 0] = 0

    return result_mlpg