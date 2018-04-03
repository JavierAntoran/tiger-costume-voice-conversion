# Now we can import our new module and call our function.
from cepstrum_dtw_utils import *
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def from_wav(file):
    x, fs = sf.read(file)

    if x.ndim is 2:
        x = x[:, 0]
    x = x.copy(order='C').astype(float)
    return x, fs


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


def f0_start_end(f0):
    nzc = np.count_nonzero(f0)
    if nzc > 0:
        nz = np.flatnonzero(f0)
        start = nz[0]
        end = nz[-1]
        if start == end:
            if np.count_nonzero(f0[0:start]):
                return (0, end)
            else:
                return (start, len(f0))
        else:
            if nz.size > 2:
                return (start, end)
    else:
        return ()


def train_samples(x, fs, ns=1, t1=400, t2=10, nth1=50, nth2=20, display=False):
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

    noise_power = np.sum(np.power(np.absolute(x[0:np.floor(ns * fs).astype(int) - 1]), 2))
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

    if display:
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


def wvec_mv_norm(samples, cutoffs, means=None, stds=None):
    cut_f0 = cutoffs[0]
    cut_sp = cutoffs[1]

    f0 = samples[:, :cut_f0]
    sp = samples[:, cut_f0:cut_sp + 1]
    ap = samples[:, cut_sp + 1:]

    if means is not None:
        f0_mean = means[:cut_f0]
        sp_mean = means[cut_f0:cut_sp + 1]
        ap_mean = means[cut_sp + 1:]
    else:
        f0_mean = np.mean(f0)
        sp_mean = np.mean(sp, axis=0)
        ap_mean = np.mean(ap)

    f0_mn = f0 - f0_mean
    sp_mn = sp - sp_mean
    ap_mn = ap - ap_mean

    if stds is not None:
        f0_std = stds[:cut_f0]
        sp_std = stds[cut_f0:cut_sp + 1]
        ap_std = stds[cut_sp + 1:]
    else:
        f0_std = np.std(f0_mn)
        sp_std = np.std(sp_mn, axis=0)
        ap_std = np.std(ap_mn)

    if f0_std == 0:
        f0_std = 1
    f0_mvn = f0_mn / f0_std

    sp_std[sp_std == 0] = 1
    sp_mvn = sp_mn / sp_std

    if ap_std == 0:
        ap_std = 1
    ap_mvn = ap_mn / ap_std

    out = np.concatenate((f0_mvn, sp_mvn, ap_mvn), axis=1)
    means = np.concatenate((np.reshape(f0_mean, 1), np.transpose(sp_mean), np.reshape(ap_mean, 1)))
    stds = np.concatenate((np.reshape(f0_std, 1), np.transpose(sp_std), np.reshape(ap_std, 1)))
    return out, means, stds


def wvec_mv_unnorm(samples, cutoffs, means, stds):
    cut_f0 = cutoffs[0]
    cut_sp = cutoffs[1]

    f0_mvn = samples[:, :cut_f0]
    sp_mvn = samples[:, cut_f0:cut_sp + 1]
    ap_mvn = samples[:, cut_sp + 1:]

    f0_mn = f0_mvn * stds[:cut_f0]
    sp_mn = sp_mvn * stds[cut_f0:cut_sp + 1]
    ap_mn = ap_mvn * stds[cut_sp + 1:]

    f0 = f0_mn + means[:cut_f0]
    sp = sp_mn + means[cut_f0:cut_sp + 1]
    ap = ap_mn + means[cut_sp + 1:]

    out = np.concatenate((f0, sp, ap), axis=1)
    return out
