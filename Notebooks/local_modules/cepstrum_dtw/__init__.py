

from __future__ import absolute_import, division, print_function
from scipy.fftpack import dct
from scipy import signal
import numpy as np

def windower(x, M, N):
  # M avance entre vetanas
  # N windowsize

  T   = x.shape[0]
  m   = np.arange(0, T-N+1, M) # comienzos de ventana
  L   = m.shape[0] # N ventanas
  ind = np.expand_dims(np.arange(0, N), axis=1) * np.ones((1,L)) + np.ones((N,1)) * m
  X   = x[ind.astype(int)]
  return X.transpose()
  
def gen_mfb_mtx(NFFT, nfilt, sample_rate):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    return fbank
  
def get_mel_cepstrum(x, fs, N_t, M_t, hamming=True, NFFT=512, nMelFilt = 24, nCeps = 12, substractMean = True):
    sample_rate = fs
    N = np.round(N_t * sample_rate).astype(int)
    M = np.round(M_t * sample_rate).astype(int)
    #% M avance entre vetanas
    #% N windowsize

    # Nwindows, samples/window
    A = windower(x, M, N)
    if hamming:
        W = np.expand_dims(np.hamming(A.shape[1]), axis=0)
        A_hamm = A * W
        A = A_hamm
    # nwindows, (NFFT/2)+1
    s = np.fft.rfft(A, n=NFFT, axis=1)
    s = abs(s)

    # nfilt, NFFT/2
    fbank = gen_mfb_mtx(NFFT, nMelFilt, sample_rate)

     # (nwindows, nMelFilt)       
    filtered_s = np.matmul(s, fbank.T)
    filtered_s = np.where(filtered_s == 0, np.finfo(float).eps, filtered_s)  # Numerical Stability
    log_filtered_s = np.log(filtered_s)


    # Nwindows, nCeps
    mfcc = dct(log_filtered_s, type=2, axis=1, norm='ortho')[:, 1 : (nCeps + 1)] # Keep 2-13
    if substractMean:
        mfcc = mfcc - np.expand_dims(np.mean(mfcc, axis=1) + 1e-8, 1)
        
    return mfcc
  
  
def matx_derivatives(X, Nderivatives=2):

    filter1 = np.array([-1.0000,   -0.7500,   -0.5000,   -0.2500,         0,    0.2500,    0.5000,   0.7500,    1.0000])
    filter2 = np.array([ 1.0000,    0.2500,   -0.2857,   -0.6071,   -0.7143,   -0.6071,   -0.2857,   0.2500,    1.0000])
    filter1 = filter1[:, None]
    filter2 = filter2[:, None]

    d1 = signal.convolve2d(X, filter1, mode='same')
    if Nderivatives is 2:
        d2 = signal.convolve2d(X, filter2, mode='same')
        X_d_dd = np.concatenate((X, d1, d2), axis=1)
    else:
        X_d_dd = np.concatenate((X, d1), axis=1)
    
    return X_d_dd
  
  
def elementwyse_l2(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if x.ndim is 1:
        x = x.reshape(-1,1)
        x_norm = (x**2).reshape(-1, 1)
    else:
        x_norm = (x**2).sum(1).reshape(-1, 1)
        
    if y is not None:
        if y.ndim is 1:
            y = y.reshape(-1, 1)
            y_norm = (y**2).reshape(1, -1)
        else:
            y_norm = (y**2).sum(1).reshape(1, -1)
    else:
        y = x
        y_norm = x_norm.reshape(1, -1)

    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return dist


def genate_Cmtx(dists):  
  sz = dists.shape
  c = dists
  d = 100000*np.ones(sz)
  #torch:
  #sz[0] = i vertical
  #sz[1] = j horizontal
  d[:, 0] = c[:, 0]
  
  cost_mult = 1+ np.sqrt((dists.shape[0]**2)+(dists.shape[1]**2))/(dists.shape[1]**2)
  
  for j in range(1, sz[1]):
    for i in range(sz[0]):

      if i is 0:
        d[i, j] = c[i,j] + d[i,j-1]
      else:
        d[i, j] = c[i,j] + np.array([d[i,j-1]*cost_mult, d[i-1,j], d[i-1,j-1]]).min()
        
  return d  


def dtw_backtracking(c, start):
  # Work in progress
  posV = np.array([start[0]])
  posH = np.array([start[1]])
  i = 0
  
  while posH[i] != 0:
#     print(posV[i],posH[i])
    if posV[i] == 0:
      opts = 1

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