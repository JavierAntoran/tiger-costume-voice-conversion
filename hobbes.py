#!/usr/bin/python
import pyaudio
import numpy as np
import thread, time
import soundfile as sf
import pyworld as pw
import pysptk
import sys
import time
import sounddevice as sd
import argparse
from test_run_mlpg import run_mlpg_regression
from src.mlpg import *
from src.regression_utils import *
from models.BaseNet import Net
from models.ff_probabilistic_mlpg import Nloss_GD, ff_mlpg

#w_ratio = 4;
#M_t = N_t/w_ratio;

CHUNK = 2048 # number of data points to read at a time
RATE = 16000 # time resolution of the recording device (Hz)
TAPE_LENGTH=5 #seconds
tape = np.zeros(RATE* TAPE_LENGTH)

f0_floor = 71.0
f0_ceil = 800.0
frame_period = 5.0

#DIO
channels_in_octave = 2.0
speed= 1
allowed_range = 0.1

q1 = -0.15
threshold = 0.85

fft_size = pw.get_cheaptrick_fft_size(RATE,f0_floor)
 
alpha=0.58
mcsize=59
order=4

def toquefrency(f0):
  #$sptk/sopr -magic 0.0 -LN -MAGIC -1.0E+10 > ${lf0_dir}/$file_id.lf0
  lf0 = np.zeros(f0.shape,dtype=float)
  magic = np.array(np.where(f0 == 0))
  imask = np.ones(f0.shape,dtype=bool)
  imask[magic] = False
  lf0[np.logical_not(imask)] = -1.0e10;
  lf0[imask] = np.log(f0[imask])
  return lf0

def tofrequency(lf0):
  #{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], x2x=SPTK['X2X'], f0=files['f0']
  f0 = np.zeros(lf0.shape,dtype=float)
  magic = np.array(np.where(lf0 == -1e10))
  imask = np.ones(lf0.shape,dtype=bool)
  imask[magic] = False
  f0[np.logical_not(imask)] = 0.0
  f0[imask] = np.exp(lf0[imask])
  return f0

def trim_zeros_frames(f0, sp, ap, eps=1e-7):
    """Remove trailling zeros frames.
    Similar to :func:`numpy.trim_zeros`, trimming trailing zeros features.
    Args:
        x (numpy.ndarray): Feature matrix, shape (``T x D``)
        eps (float): Values smaller than ``eps`` considered as zeros.
    Returns:
        numpy.ndarray: Trimmed 2d feature matrix, shape (``T' x D``)
    Examples:
        >>> import numpy as np
        >>> from nnmnkwii.preprocessing import trim_zeros_frames
        >>> x = np.random.rand(100,10)
        >>> y = trim_zeros_frames(x)
    """

    T, D = sp.shape
    s = np.sum(np.abs(sp), axis=1)
    #print(s)
    s[s < eps] = 0.0
    lz = len(s) - len(np.trim_zeros(s,'f'));
    tz = len(s) - len(np.trim_zeros(s,'b'));
    f0 = f0[lz:len(s)-tz]
    sp = sp[lz:(len(s)-tz),:]
    ap = ap[lz:(len(s)-tz),:]
    
    return lz, tz, f0, sp, ap

def input_thread(L):
    raw_input()
    L.append(None)

def acquire_audio():
    print("Recording time: %d s" % (TAPE_LENGTH))
    raw_input('Press a key to start recording....')
    print('Recording ....')
    p = pyaudio.PyAudio()  # start the PyAudio class
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
		    input_device_index=4, 
                    frames_per_buffer=CHUNK)  # uses default input device
    L = []
    thread.start_new_thread(input_thread, (L,))
    taps = 0
    end_time = time.time() + TAPE_LENGTH
    while 1 and taps < np.floor(RATE*TAPE_LENGTH/CHUNK):
        if L: break
        tape[taps*CHUNK:taps*CHUNK+CHUNK] = np.fromstring(stream.read(CHUNK), dtype=np.int16)
        taps = taps+1
        sys.stdout.write('\r')
        sys.stdout.write("Time remains %d s" % (end_time - time.time()))
        sys.stdout.flush()
    # close the stream gracefully
    print()
    stream.stop_stream()
    stream.close()
    p.terminate()



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TigerCostume')

    parser.add_argument('-dn','--dataset-name', action="store")
    parser.add_argument('-ts', action="store_true")
    parser.add_argument('-ts_num', action="store", type=int)
    parser.add_argument('--harvest', action="store_true", default=False)

    opts = parser.parse_args()

    out_filename = ''
    if not opts.dataset_name:
        acquire_audio()
        tape = tape/np.max(np.abs(tape))
        sf.write("test.wav", tape/2**15, RATE)
        print("Audio acquired")
        x=tape/2**15
        x=np.trim_zeros(x)
        del tape
        out_filename += 'acq_'
    else:
        print("Loading file " + './audios/'+opts.dataset_name+'_'+str(opts.ts_num)+'.wav')
        x,_   = sf.read('./audios/'+opts.dataset_name+'_'+str(opts.ts_num)+'.wav')
    
        if x.ndim is 2:
            x = x[:, 0]
        x = x.copy(order='C').astype(float)
        x=np.trim_zeros(x)
        out_filename += opts.dataset_name+'_'+str(opts.ts_num)+'_'
        
    if opts.harvest:    
        print("Begin harvest ...")
        f0_x, tp_x = pw.harvest(x, RATE,f0_floor, f0_ceil, frame_period)
        out_filename += 'harvest'
    else :
        print("Begin stonemask ...")
        f0_x, tp_x = pw.dio(x, RATE,f0_floor, f0_ceil, channels_in_octave,  frame_period, speed, allowed_range)
        f0_x = pw.stonemask(x, f0_x, tp_x, RATE)
        out_filename += 'dio'
    
    print("Begin cheaptrick ...")
    sp_x = pw.cheaptrick(x, f0_x, tp_x, RATE, q1, f0_floor, fft_size)
    print("Begin d4c ...")
    ap_x = pw.d4c(x, f0_x, tp_x, RATE, threshold, fft_size)

    lz, tz,f0_x, sp_x, ap_x = trim_zeros_frames(f0_x,sp_x, ap_x, 0.7)
    
    uv = (f0_x == 0).astype(int)
   
    print("Begin f0 transform ...")
    lf0_x = toquefrency(f0_x)
    print("Begin sp transform ...")
    mgc_x = pysptk.conversion.sp2mc(sp_x, order = mcsize, alpha=alpha)
    print("Begin ap transform ...")
    bap_x = pysptk.conversion.sp2mc(ap_x, order = mcsize, alpha=alpha)

    statsdir = 'model_saves/ST_STATS_mlpg.npy'
    savedir = 'model_saves/theta_best_mlpg.dat'

    #hash_md5 = hl.md5()
    #with open(savedir, "rb") as f:
    #    for chunk in iter(lambda: f.read(4096), b""):
    #        hash_md5.update(chunk)
    #print('theta_best_mlpg: ' + hash_md5.hexdigest())

    feature_x = np.concatenate((lf0_x.reshape(len(lf0_x),1), mgc_x),1)
    feature_x[feature_x[:,0] == -1e10] = 0

    print("Neural processing ...")
    features_y = run_mlpg_regression(feature_x, statsdir = statsdir, savedir=savedir)

    lf0_y = np.array(features_y[:,0], order='C')
    lf0_y[uv==1] = -1e10

    mgc_y = np.array(features_y[:,1:], order='C')

    print("Undo f0 transformation ...")
    f0_y = tofrequency(lf0_y)
    print("Undo bap transformation ...")
    ap_y = pysptk.conversion.mc2sp(bap_x.astype(np.float64), alpha=alpha, fftlen = fft_size)
    print("Undo mgc transformation ...")
    sp_y = pysptk.conversion.mc2sp(mgc_y.astype(np.float64),alpha=alpha, fftlen = fft_size)
    
    out_filename += '.wav'
    y = pw.synthesize(f0_y, sp_y, ap_y, RATE, frame_period)
    sf.write(out_filename, y, RATE)
    print("DONE")
