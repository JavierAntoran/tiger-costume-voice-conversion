from __future__ import print_function
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import sys

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
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

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def wvec_mv_norm(samples, cutoffs, means=None, stds=None):
    cut_f0 = cutoffs[0]
    cut_sp = cutoffs[1]
    
    f0 = samples[:, :cut_f0]
    sp = samples[:, cut_f0:cut_sp+1]
    ap = samples[:, cut_sp+1:]
    
    if means is not None:
        f0_mean = means[:cut_f0]
        sp_mean = means[cut_f0:cut_sp+1]
        ap_mean = means[cut_sp+1:]
    else:
        f0_mean = np.mean(f0)
        sp_mean = np.mean(sp, axis=0)
        ap_mean = np.mean(ap)
      
    f0_mn = f0 - f0_mean
    sp_mn = sp - sp_mean
    ap_mn = ap - ap_mean
    
    if stds is not None:
        f0_std = stds[:cut_f0]
        sp_std = stds[cut_f0:cut_sp+1]
        ap_std = stds[cut_sp+1:]
    else:
        f0_std = np.std(f0_mn)
        sp_std = np.std(sp_mn, axis=0)
        ap_std = np.std(ap_mn)
    
    if f0_std == 0:
      f0_std = 1
    f0_mvn = f0_mn / f0_std
    
    sp_std[sp_std==0] = 1
    sp_mvn = sp_mn / sp_std
    
    if ap_std == 0:
      ap_std = 1
    ap_mvn = ap_mn / ap_std
    
    out = np.concatenate((f0_mvn, sp_mvn, ap_mvn), axis=1)
    means = np.concatenate((np.reshape(f0_mean,1), np.transpose(sp_mean), np.reshape(ap_mean,1)))
    stds = np.concatenate((np.reshape(f0_std,1), np.transpose(sp_std), np.reshape(ap_std,1)))
    return out, means, stds

def wvec_mv_unnorm(samples, cutoffs, means, stds):
    cut_f0 = cutoffs[0]
    cut_sp = cutoffs[1]
    
    f0_mvn = samples[:, :cut_f0]
    sp_mvn = samples[:, cut_f0:cut_sp+1]
    ap_mvn = samples[:, cut_sp+1:]
    
    
    f0_mn = f0_mvn * stds[:cut_f0]
    sp_mn = sp_mvn * stds[cut_f0:cut_sp+1]
    ap_mn = ap_mvn * stds[cut_sp+1:]
    
    f0 = f0_mn + means[:cut_f0]
    sp = sp_mn + means[cut_f0:cut_sp+1]
    ap = ap_mn + means[cut_sp+1:]
    
    out = np.concatenate((f0, sp, ap), axis=1)
    return out
  
def feature_mv_norm(samples, cutoffs, means=None, stds=None):
    cut_f0 = cutoffs[0]
    cut_sp = cutoffs[1]
    
    f0 = samples[:, 0:cut_f0]
    sp = samples[:, cut_f0:cut_sp+1]
    
    uv = np.zeros(f0.shape, dtype=bool)
    uv[f0 is 0.0] = True
    
    if means is not None:
        f0_mean = means[:cut_f0]
        sp_mean = means[cut_f0:cut_sp+1]
    else:
        f0_mean = np.mean(f0[np.logical_not(uv)])
        sp_mean = np.mean(sp, axis=0)
    
    f0_mn = f0 - f0_mean
    f0_mn[uv] = 0.0;
    
    sp_mn = sp - sp_mean
    
    if stds is not None:
        f0_std = stds[:cut_f0]
        sp_std = stds[cut_f0:cut_sp+1]
    else:
        f0_std = np.std(f0_mn[np.logical_not(uv)])
        sp_std = np.std(sp_mn, axis=0)
    
    if f0_std == 0:
      f0_std = 1
    f0_mvn = f0_mn / f0_std
    
    sp_std[sp_std==0] = 1
    sp_mvn = sp_mn / sp_std
    
       
    out = np.concatenate((f0_mvn, sp_mvn), axis=1)
    means = np.concatenate((np.reshape(f0_mean,1), np.transpose(sp_mean)))
    stds = np.concatenate((np.reshape(f0_std,1), np.transpose(sp_std)))
    return out, means, stds, uv

def feature_mv_unnorm(samples, cutoffs, means, stds, uv):
    cut_f0 = cutoffs[0]
    cut_sp = cutoffs[1]
    
    f0_mvn = samples[:, :cut_f0]
    sp_mvn = samples[:, cut_f0:cut_sp+1]
    
    
    f0_mn = f0_mvn * stds[:cut_f0]
    sp_mn = sp_mvn * stds[cut_f0:cut_sp+1]
    
    f0 = f0_mn + means[:cut_f0]
    sp = sp_mn + means[cut_f0:cut_sp+1]
    
    f0[uv] = 0.0
    
    out = np.concatenate((f0, sp), axis=1)
    return out
  
def readStats(file):
  ds_stats = np.load(file)
  Xmeans = ds_stats[0,:]
  Xstds = ds_stats[1,:]
  Xuv = ds_stats[2,:]
  Tmeans = ds_stats[3,:]
  Tstds = ds_stats[4,:]
  Tuv = ds_stats[5,:]
  return Xmeans, Xstds, Xuv, Tmeans, Tstds, Tuv
  