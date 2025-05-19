import numpy as np
import scipy.io
from plots import plot_neigungsdaten, plot_delta_t

###! Loading data
mat_contents = scipy.io.loadmat('data/Neigung.mat')
# Neigung-x, Neigung-y, Temperatur, Zeit in s || Abtastinterval: 120s
neigung_zeitreihe = np.array(mat_contents['N']) 


if __name__ == "__main__":
    # Anzeigen der rohen Neigungsdaten
    plot_neigungsdaten(neigung_zeitreihe)
    
    # Auffinden von Datenlücken
    # --> dazu Berechnen der Differenzen der Zeitstempel (dort wo Differenz > 120s)
    plot_delta_t(neigung_zeitreihe[:, 3])