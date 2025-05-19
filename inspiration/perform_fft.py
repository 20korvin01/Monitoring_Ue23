# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:21:12 2020

@author: Scheider
"""

def perform_fft(ts_red,F_s):
    #Input:     ts_red - trendbereinigte Zeitreihe (Vektor)
    #           F_s - Samplingrate
    #Output:    f0 - Frequenzen (Vektor)
    #           P1 - zu f0 gehörende Amplituden (Vektor), für einseitiges Amplitudendiagramm (vgl. Matlab)
    
    #Länge der Zeitreihe anpassen
    if len(ts_red)%2==0:
        tt1 = np.asarray(ts_red[:,0])
    else:
        tt1 = np.asarray(ts_red[0:len(ts_red)-1,0])
    n = len(tt1)-1

    FouT1 = np.fft.fft(tt1)
    P2 = np.abs(FouT1/n)
    stp_ind= int(n/2)
    P1 = np.array(P2[0:stp_ind])
    P1[1:stp_ind-1] = 2*P1[1:stp_ind-1]

    f0 = np.array(F_s*np.arange(0,(n/2)-1)/n)
    return f0, P1