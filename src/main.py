import numpy as np
import scipy.io
from plots import plot_neigungsdaten, plot_delta_t
import matplotlib.pyplot as plt

def linear_interpolation(timestamps, values, t_new):
    """
    Interpoliert die Werte auf die neuen Zeitstempel t_new
    mithilfe der linearen Interpolation.
    """
    values_interp = np.zeros(len(t_new))
    for i in range(len(t_new)):
        # Finde den Index des nächsten Zeitstempels
        idx = np.searchsorted(timestamps, t_new[i])
        
        if idx == 0:
            values_interp[i] = values[0]
        else:
            # Interpolation zwischen den benachbarten Werten
            t1 = timestamps[idx - 1]
            t2 = timestamps[idx]
            v1 = values[idx - 1]
            v2 = values[idx]
            values_interp[i] = v1 + (v2 - v1) * (t_new[i] - t1) / (t2 - t1)
            
    return values_interp


if __name__ == "__main__":   
    
    ### AUFGABE 0 #####################################################################################
    """Laden und Darstellen der Messwerte"""
    # Laden der Messwerte
    mat_contents = scipy.io.loadmat('data/Neigung.mat')
    # Neigung-x, Neigung-y, Temperatur, Zeit in s || Abtastinterval: 120s
    neigung_zeitreihe = np.array(mat_contents['N'])
    neigung_x = neigung_zeitreihe[:, 0]
    neigung_y = neigung_zeitreihe[:, 1]
    temperatur = neigung_zeitreihe[:, 2]
    t = neigung_zeitreihe[:, 3]
    # Plotten der Messwerte
    # plot_neigungsdaten(neigung_zeitreihe)
       
    
    ### AUFGABE 1 #####################################################################################
    """Aufbereitung der Messwerte"""
    ## 1.1 Anschauliche Darstellung der Datenlücken ------------------------------------------------ ##
    # --> dazu Berechnen der Differenzen der Zeitstempel (dort wo Differenz > 120s)
    # plot_delta_t(neigung_zeitreihe[:, 3])
    
    ## 1.2 Füllen der Datenlücken durch lineare Interpolation -------------------------------------- ##
    # Abtastintervall
    dT = 120
    # Differenzen der Zeitstempel
    delta_t = np.diff(t)
    # Start- und Endzeit
    t_start = t[0]
    t_end = t[-1]
    # neue Zeitreihe
    t_new = np.arange(t_start, t_end, dT)
    # lineare Interpolation der Neigungsdaten
    neigung_x_interp = linear_interpolation(t, neigung_x, t_new)
    neigung_y_interp = linear_interpolation(t, neigung_y, t_new)
    temperatur_interp = linear_interpolation(t, temperatur, t_new)
    
    
    