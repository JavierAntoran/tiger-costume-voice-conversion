import pyworld as pw
import soundfile as sf
import sys



file = sys.argv[1]

x, fs = sf.read(file)

if x.ndim is 2:
    x = x[:, 0].squeeze()
x = x.copy(order='C')

f0, sp, ap = pw.wav2world(x, fs)
y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)

sf.write("reconstruct" + file, y, fs)

print(f0)
