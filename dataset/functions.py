import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.fftpack import dct



def from_wav(file):
    x, fs = sf.read(file)
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
            chunks.append(indice[i]);
        elif (window_power < noise_th2 and sflag == 1):
            clip[indice[i] + N1] = window_power
            sflag = 0
            chunks.append(indice[i] + N1)
        wp[indice[i]] = window_power;

    # fig, ax1 = plt.subplots()
    # ax1.plot(t, x, 'b-')
    # ax1.set_xlabel('time (s)')
    # # Make the y-axis label, ticks and tick labels match the line color.
    # ax1.set_ylabel('signal', color='b')
    # ax1.tick_params('y', colors='b')
    #
    # ax2 = ax1.twinx()
    # ax2.plot(t, clip, 'r.')
    # ax2.set_ylabel('wp', color='r')
    # ax2.tick_params('y', colors='r')
    #
    # fig.tight_layout()
    # plt.show()

    print(len(chunks))
    max_chunk_dur = np.max(np.subtract(chunks[1:2:], chunks[0:2:]))

    train_samples = np.zeros(((int)(len(chunks) / 2), max_chunk_dur))

    ti = 0;
    for i in np.arange(0, len(chunks), 2):
        chunk_dur = chunks[i + 1] - chunks[i];
        chunk_diff = max_chunk_dur - chunk_dur;
        if (np.mod(chunk_diff, 2) == 0):
            off_s = (int)(chunk_diff / 2);
            off_e = off_s;
        else:
            off_s = np.floor(chunk_diff / 2).astype(int);
            off_e = off_s + 1;
        train_samples[[ti], :] = x[chunks[i] - off_s:chunks[i + 1] + off_e];
        # soundsc(s(chunks(i):chunks(i+1)),fs)
        # pause
        ti = ti + 1

    return train_samples


def silence_filter(x, frame_period, fs, th=0.1):
    frame_samples = np.round(fs * frame_period / 1000).astype(int);

    N = (6 * frame_samples).astype(int);
    M = (2 * frame_samples).astype(int);

    x_w = windower(x, M, N);

    x_power = np.apply_along_axis(np.power, 0, np.absolute(x_w), 2)
    x_power = np.apply_along_axis(np.sum, 1, x_power)

    speech_th = np.array(np.where(x_power > th))
    speech_index = (speech_th * M).astype(int)
    noise_th = np.array(np.where(x_power <= th))
    noise_index = (noise_th * M).astype(int)
    # speech_length = np.transpose(np.concatenate((x_th*M, np.ones(x_th.shape)*M), axis=0))
    # speech_index = tuple(map(int,speech_index))
    x_fil = np.concatenate([x[offset:(offset + M)] for offset in speech_index[0]])
    return x_fil, speech_index, noise_index, M


def windower(x, M, N):
    # M avance entre vetanas
    # N windowsize

    T = x.shape[0]
    m = np.arange(0, T - N + 1, M)  # comienzos de ventana
    L = m.shape[0]  # N ventanas
    ind = np.expand_dims(np.arange(0, N), axis=1) * np.ones((1, L)) + np.ones((N, 1)) * m
    X = x[ind.astype(int)]
    return X.transpose()


def gen_mfb_mtx(NFFT, nfilt, sample_rate):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank


def get_mel_cepstrum(x, fs, N_t, M_t, hamming=True, NFFT=512, nMelFilt=24, nCeps=12, substractMean=True):
    sample_rate = fs
    N = np.round(N_t * sample_rate).astype(int)
    M = np.round(M_t * sample_rate).astype(int)
    # % M avance entre vetanas
    # % N windowsize

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
    mfcc = dct(log_filtered_s, type=2, axis=1, norm='ortho')[:, 1: (nCeps + 1)]  # Keep 2-13
    if substractMean:
        mfcc = mfcc - np.expand_dims(np.mean(mfcc, axis=1) + 1e-8, 1)

    return mfcc



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

def f0_start_end(f0):
  nzc = np.count_nonzero(f0)
  if nzc > 0:
    nz = np.flatnonzero(f0)
    start = nz[0]
    end = nz[-1]
    if start == end:
      if np.count_nonzero(f0[0:start]):
        return (0,end)
      else:
        return (start, len(f0))
    else:
      if nz.size > 2:
        return (start, end)
  else:
    return ()