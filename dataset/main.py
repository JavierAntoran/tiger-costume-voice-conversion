import os
import numpy as np
from scipy import signal, misc
import functions



basePath = './'
print(os.listdir(basePath))

print('Loading train_data')
ST = np.load(basePath+"/train_data/ST_full.npy")
print(ST.shape)
STU = np.unique(ST,axis=0)
del ST
print(STU.shape)
STS = np.split(STU,2,axis=1)
del STU
S = STS[0]
T = STS[1]
del STS
print(S.shape,T.shape)
print('Saving Source')
np.save(basePath+"/train_data/S_clean_local.npy",S)
print('Saving Target')
np.save(basePath+"/train_data/T_clean_local.npy",T)
