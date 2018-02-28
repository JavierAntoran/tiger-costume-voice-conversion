import numpy as np
from matplotlib import pyplot as plt
from src.dtw_lib import *

# code for dynamic time warping on time series

s = np.array([0, 1, 3, 5, 8, 6, 4])
t = np.array([0, 1, 2, 1, 4, 7, 5, 4, 4])

D = elementwyse_l2(s, t)
C = genate_Cmtx(D)
start = [6,8]
posV, posH = dtw_backtracking(C, start)

plt.figure()
plt.imshow(D)
plt.gca().invert_yaxis()
plt.figure()
plt.imshow(C)
plt.plot(posH, posV, 'r--o')
plt.gca().invert_yaxis()
plt.show()
