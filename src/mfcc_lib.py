# Cepstrum lib
import numpy as np
from scipy.fftpack import dct

def windower(x, M, N):
    # M avance entre vetanas
    # N windowsize

    T = x.shape[0]
    m = np.arange(0, T - N - 1, M)  # comienzos de ventana
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
    N = np.round(0.03 * sample_rate).astype(int)
    M = np.round(0.01 * sample_rate).astype(int)
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