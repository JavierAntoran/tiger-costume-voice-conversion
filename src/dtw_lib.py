import numpy as np


def elementwyse_l2(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if x.ndim is 1:
        x = x.reshape(-1, 1)
        x_norm = (x ** 2).reshape(-1, 1)
    else:
        x_norm = (x ** 2).sum(1).reshape(-1, 1)

    if y is not None:
        if y.ndim is 1:
            y = y.reshape(-1, 1)
            y_norm = (y ** 2).reshape(1, -1)
        else:
            y_norm = (y ** 2).sum(1).reshape(1, -1)
    else:
        y = x
        y_norm = x_norm.reshape(1, -1)

    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return dist


def genate_Cmtx(dists):
    sz = dists.shape
    c = dists
    d = 100000 * np.ones(sz)
    # torch:
    # sz[0] = i vertical
    # sz[1] = j horizontal
    d[:, 0] = c[:, 0]

    for j in range(1, sz[1]):
        for i in range(sz[0]):

            if i is 0:
                d[i, j] = c[i, j] + d[i, j - 1]
            else:
                d[i, j] = c[i, j] + np.array([d[i, j - 1], d[i - 1, j], d[i - 1, j - 1]]).min()

    return d


def dtw_backtracking(c, start):
    # Work in progress
    posV = np.array([start[0]])
    posH = np.array([start[1]])
    i = 0

    while posH[i] != 0:

        if posV[i] == 0:
            opts = 2

        else:
            opt1 = c[posV[i] - 1, posH[i]]
            opt2 = c[posV[i], posH[i] - 1]
            opt3 = c[posV[i] - 1, posH[i] - 1]

            aa = np.array([opt1, opt2, opt3], dtype=np.float32)
            opts = np.argmin(aa)

        nextV = posV[i] - (opts == 0 or opts == 2)
        nextH = posH[i] - (opts == 1 or opts == 2)

        posV = np.append(posV, nextV)
        posH = np.append(posH, nextH)

        i += 1

    return posV, posH