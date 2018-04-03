import numpy as np

filter1 = np.array([-1.0000,   -0.7500,   -0.5000,   -0.2500,         0,    0.2500,    0.5000,   0.7500,    1.0000])
filter2 = np.array([ 1.0000,    0.2500,   -0.2857,   -0.6071,   -0.7143,   -0.6071,   -0.2857,   0.2500,    1.0000])


def get_mtx_deltas(X, filter1, filter2):
    Nwin = X.shape[0]
    W1 = get_delta_mtx(filter1, Nwin)
    W2 = get_delta_mtx(filter2, Nwin)
    delta1 = np.dot(W1, X)
    delta2 = np.dot(W2, X)

    out = np.concatenate((X, delta1, delta2), axis=1)
    return out


def get_delta_mtx(the_filter, Nwin):
    # returns Nwin by Nwin matrix W1. W1 * base_features = delta1
    Fsize = len(the_filter)
    T_pad = (Fsize - 1)
    Nwin_pad = Nwin + T_pad
    pad_s = int(T_pad / 2)
    pad_e = int(T_pad / 2)  # + Nwin_pad%2
    # print('padding start, end:', pad_s, pad_e)
    # Generate Filter
    W1 = np.zeros((Nwin, Nwin_pad))
    for i in range(Nwin_pad - Fsize + 1):
        W1[i, i:i + Fsize] = the_filter
    W1 = W1[:, pad_s:]
    W1 = W1[:, :-pad_e]
    return W1


def my_mlpg(features, dims, W1, W2, Beta):
    # Beta is precision of each dim: inverse of var. Pls sanitize before introducing.
    Nwins = features.shape[0]
    y = features[:, 0:dims]
    delta1_y = features[:, dims:2 * dims]
    delta2_y = features[:, 2 * dims:3 * dims]
    W0 = np.eye(Nwins)
    ml_params = np.zeros((Nwins, dims))

    for d in range(dims):
        # Variance of each parameter along dimension d
        D0 = Beta[:, d] * np.eye(Nwins)
        D1 = Beta[:, dims + d] * np.eye(Nwins)
        D2 = Beta[:, 2 * dims + d] * np.eye(Nwins)
        # Mean output of each parameter from neural net
        U0 = y[:, d]
        U1 = delta1_y[:, d]
        U2 = delta2_y[:, d]
        # Compute first term
        wdw0 = D0
        wdw1 = np.transpose(W1).dot(D1).dot(W1)
        wdw2 = np.transpose(W2).dot(D2).dot(W2)
        WDW = wdw0 + wdw1 + wdw2
        # computer second term
        wdu0 = D0.dot(U0)
        wdu1 = np.transpose(W1).dot(D1).dot(U1)
        wdu2 = np.transpose(W2).dot(D2).dot(U2)
        WDU = wdu0 + wdu1 + wdu2

        vec_params = np.dot(np.linalg.inv(WDW), WDU)
        ml_params[:, d] = vec_params

    return ml_params