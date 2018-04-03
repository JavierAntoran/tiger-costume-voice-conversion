import os
import sys
from dataset import functions as df
import plotly
plotly.tools.set_credentials_file(username='almul0', api_key='JOYPW5io1T9oG6cxpSv2')
import plotly.plotly as py
import plotly.graph_objs as go
import pyworld as pw
import numpy as np
from scipy import signal, misc

basePath = os.path.dirname(os.path.realpath(__file__));

file_javi = basePath + '/../sources/javi_uno.wav'
file_alberto = basePath + '/../sources/alberto_uno.wav'
file_raquel = basePath + '/../sources/raquel_uno_mod.wav'

source_file = file_javi
target_file = file_alberto


#Harvest parameters
N_t = 0.020
w_ratio = 4;
M_t = N_t/w_ratio;

f0_floor = 71.0
f0_ceil = 800.0
frame_period = M_t*1000;

q1 = -0.15
threshold = 0.85

dec_factor = 2


def main():
    x, fs = df.from_wav(source_file)
    y, fs = df.from_wav(target_file)

    fft_size = pw.get_cheaptrick_fft_size(fs, f0_floor)
    fs_ds = fs / dec_factor;

    print("N_t (Long ventana): %.3f s" % (N_t))
    print("M_t (Avance ventana): %.3f s" % (M_t))

    print("Numero de ventanas(f0) por ventana (N): %d " % (w_ratio))
    print("Periodo de ventana para f0: %.3f s" % (frame_period / 1000))
    print("Tamaño FFT: %.3f s" % (fft_size))

    print("Factor de diezmado: %d" % (dec_factor))
    print("Fs: %.2f Hz to %.2f Hz" % (fs, fs_ds))

    if dec_factor > 1:
        x = signal.decimate(x, dec_factor, ftype='fir');
    ts_s = df.train_samples(x, fs_ds)
    del x

    if dec_factor > 1:
        y = signal.decimate(y, dec_factor, ftype='fir');
    ts_t = df.train_samples(y, fs_ds)
    del y

    batches = range(ts_s.shape[0])
    batches = [1]

    # print(batches)

    for ts_s_i in batches:
        s_data = np.empty((0, fft_size + 3))  # feature_sbatches = [0,1,2,3,4,5,6,7].shape[1]))
        t_data = np.empty((0, fft_size + 3))  # feature_s.shape[1]))
        tmp_s = silence_filter(ts_s[ts_s_i, :], frame_period, fs_ds)[0]

        f0_s, tp_s = pw.harvest(tmp_s, fs_ds, f0_floor, f0_ceil, frame_period)

        avance = int(frame_period * (fs_ds / 1000))

        if np.count_nonzero(f0_s) > 0:
            print("Processing batch %d..." % (ts_s_i))
            f0_s_se = f0_start_end(f0_s)
            sp_s = pw.cheaptrick(tmp_s, f0_s, tp_s, fs_ds, q1, f0_floor, fft_size)
            ap_s = pw.d4c(tmp_s, f0_s, tp_s, fs_ds, threshold, fft_size)

            f0_s = f0_s[f0_s_se[0]:f0_s_se[1]]
            sp_s = sp_s[f0_s_se[0]:f0_s_se[1], :]
            ap_s = ap_s[f0_s_se[0]:f0_s_se[1], :]

            feature_s = np.concatenate((f0_s.reshape(len(f0_s), 1), sp_s, ap_s), 1)

            tmp_s = tmp_s[int(f0_s_se[0] * avance):int(f0_s_se[1] * avance)]

            tmp_s_cep = get_mel_cepstrum(tmp_s, fs_ds, N_t, M_t, hamming=True, NFFT=512, nMelFilt=24, nCeps=13,
                                         substractMean=True)

            batches_t = range(ts_t.shape[0])
            batches_t = [3]

            for ts_t_i in batches_t:
                sys.stdout.flush()
                sys.stdout.write('\r')
                sys.stdout.write("Processing target batch %d..." % (ts_t_i))
                tmp_t = silence_filter(ts_t[ts_t_i, :], frame_period, fs_ds)[0]

                f0_t, tp_t = pw.harvest(tmp_t, fs_ds, f0_floor, f0_ceil, frame_period)

                if np.count_nonzero(f0_t) > 0:
                    f0_t_se = f0_start_end(f0_t)
                    sp_t = pw.cheaptrick(tmp_t, f0_t, tp_t, fs_ds, q1, f0_floor, fft_size)
                    ap_t = pw.d4c(tmp_t, f0_t, tp_t, fs_ds, threshold, fft_size)

                    f0_t = f0_t[f0_t_se[0]:f0_t_se[1]]
                    sp_t = sp_t[f0_t_se[0]:f0_t_se[1], :]
                    ap_t = ap_t[f0_t_se[0]:f0_t_se[1], :]

                    feature_t = np.concatenate((f0_t.reshape(len(f0_t), 1), sp_t, ap_t), 1)

                    tmp_t = tmp_t[int(f0_t_se[0] * avance):int(f0_t_se[1] * avance)]

                    tmp_t_cep = get_mel_cepstrum(tmp_t, fs_ds, N_t, M_t, hamming=True, NFFT=512, nMelFilt=24, nCeps=13,
                                                 substractMean=True)

                    D = elementwyse_l2(tmp_s_cep, tmp_t_cep)
                    C = genate_Cmtx(D)
                    start = [np.argmin(C[:, -1]), C.shape[1] - 1]
                    posV, posH = dtw_backtracking(C, start)

                    for idx_track in range(len(posV) - 1, -1, -1):
                        window_stack_s = feature_s[posV[idx_track]:int(posV[idx_track] + w_ratio), :]
                        window_stack_t = feature_t[posH[idx_track]:int(posH[idx_track] + w_ratio), :]
                        if (window_stack_s.shape == window_stack_t.shape):
                            s_data = np.append(s_data, window_stack_s, axis=0)
                            t_data = np.append(t_data, window_stack_t, axis=0)
                else:
                    print("Target batch %d discarded because no f0 detected" % (ts_s_t))
        else:
            print("Batch %d discarded because no f0 detected" % (ts_s_i))
        print("\nSaving data...")
        st_data = np.concatenate([s_data, t_data], axis=1)


if __name__ == "__main__":
    # execute only if run as a script
    main()