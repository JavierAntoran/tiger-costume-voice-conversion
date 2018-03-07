# sample DTW with mel cepstrum
from src.utils import *
from src.dtw_lib import *
from src.mfcc_lib import *

s, fs = from_wav('audio1_s.wav')
t1, fs = from_wav('audio1_t1.wav')
sample_rate = fs
N_t = 0.03
M_t = 0.01

mfcc_s = get_mel_cepstrum(s, fs, N_t, M_t, hamming=True, NFFT=512, nMelFilt = 24, nCeps = 12, substractMean = True)
mfcc_t1 = get_mel_cepstrum(t1, fs, N_t, M_t, hamming=True, NFFT=512, nMelFilt = 24, nCeps = 12, substractMean = True)

D = elementwyse_l2(mfcc_s, mfcc_t1)
C = genate_Cmtx(D)
start = [C.shape[0] - 1, C.shape[1] - 1]
posV, posH = dtw_backtracking(C, start)

plt.figure()
plt.imshow(D, cmap='jet')
plt.gca().invert_yaxis()

plt.figure()
plt.imshow(C, cmap='jet')
plt.plot(posH, posV, 'r--')
plt.gca().invert_yaxis()
