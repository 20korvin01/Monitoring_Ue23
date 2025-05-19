import numpy as np
import scipy.io

###! Loading data
mat_contents = scipy.io.loadmat('data/Neigung.mat')
# Neigung-x, Neigung-y, Temperatur, Zeit in s || Abtastinterval: 120s
neigung_zeitreihe = np.array(mat_contents['N']) 


print(neigung_zeitreihe)