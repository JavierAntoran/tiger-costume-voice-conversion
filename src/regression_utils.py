from __future__ import absolute_import, division, print_function
import soundfile as sf
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
import torch.nn as nn


def from_wav(file):
    x, fs  = sf.read(file)

    if x.ndim is 2:
        x = x[:, 0]
    x = x.copy(order='C').astype(float)
    return x,fs


def train_samples(x, fs, t1=400, t2=10, nth1=50, nth2=20):
    """Returns chunks of train samples from signal

    Args:
        x: Complete signal
        fs: Sample frequency
        t1: Duration of window (ms)
        t2: Slide length (ms)
        nth1: Upper threshold
        nth2: Bottom threshold


    Returns:
        Returns train samples of equal length

    """

    t = np.divide(np.arange(0, x.shape[0]), fs)

    L1 = x.shape[0]  # Signal length
    # T1 = 600*1e-3 # Window length
    # T2 = 20*1e-3 # Step lenght
    T1 = t1 * 1e-3  # Window length
    T2 = t2 * 1e-3  # Step lenght
    N1 = np.floor(T1 * fs).astype(int)  # Samples per window
    D1 = np.floor(T2 * fs).astype(int)  # Samples per step
    indice = np.arange(1, L1, D1, dtype=np.int32)  # Start indexes for windowing

    noise_power = np.sum(np.power(np.absolute(x[0:np.floor(fs).astype(int) - 1]), 2))
    noise_th1 = nth1 * noise_power
    noise_th2 = nth2 * noise_power
    sflag = 0
    clip = np.zeros(x.shape)
    wp = np.zeros(x.shape)
    chunks = [];

    for i in np.arange(1, indice.shape[0] - 2):
        fin = np.min([indice[i] + N1, x.shape[0]]).astype(int)
        window_power = np.sum(np.power(np.absolute(x[indice[i]:fin]), 2))
        if (window_power > noise_th1 and sflag == 0):
            sflag = 1
            clip[indice[i]] = window_power
            chunks.append(indice[i])
        elif (window_power < noise_th2 and sflag == 1):
            clip[indice[i] + N1] = window_power
            sflag = 0
            chunks.append(indice[i] + N1)
        wp[indice[i]] = window_power

    fig, ax1 = plt.subplots()
    ax1.plot(t, x, 'b-')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('signal', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(t, clip, 'r.')
    ax2.set_ylabel('wp', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()

    max_chunk_dur = np.max(np.subtract(chunks[1:2:], chunks[0:2:]))

    train_samples = np.zeros(((int)(len(chunks) / 2), max_chunk_dur))

    ti = 0
    for i in np.arange(0, len(chunks), 2):
        chunk_dur = chunks[i + 1] - chunks[i]
        chunk_diff = max_chunk_dur - chunk_dur
        if (np.mod(chunk_diff, 2) == 0):
            off_s = (int)(chunk_diff / 2)
            off_e = off_s
        else:
            off_s = np.floor(chunk_diff / 2).astype(int)
            off_e = off_s + 1
        train_samples[[ti], :] = x[chunks[i] - off_s:chunks[i + 1] + off_e]
        # soundsc(s(chunks(i):chunks(i+1)),fs)
        # pause
        ti = ti + 1

    return train_samples


suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return ((nb_samples + (-nb_samples % batch_size)) / batch_size)  # roundup division
    else:
        return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size: (i + 1) * batch_size]


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()



#     def extract(self, x, train=False):
#         x, = to_variable(var=(x,), volatile=True, cuda=self.cuda)

#         _, e = self.model(x)
#         return e.data