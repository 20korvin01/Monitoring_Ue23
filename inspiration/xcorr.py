# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:20:33 2020

@author: Scheider
"""

def xcorr(x_red, y_red, normed=True, maxlags=None):
        #Input:     x_red - trendbereinigte Zeitreihe 1
        #           y_red - trendbereinigte Zeitreihe 2
        #Output:    lags - tau (zeitliche Verschiebung) 
        #           c - Kreuzkorrelation (Wertebereich -1/+1)
        
        Nx = len(x_red)
        if Nx != len(y_red):
            raise ValueError('x and y must be equal length')

        c = np.correlate(x_red, y_red, mode=2)

        if normed:
            c /= np.sqrt(np.dot(x_red, x_red) * np.dot(y_red, y_red))

        if maxlags is None:
            maxlags = Nx - 1

        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maglags must be None or strictly '
                             'positive < %d' % Nx)

        lags = np.arange(-maxlags, maxlags + 1)
        c = c[Nx - 1 - maxlags:Nx + maxlags]

        return lags, c #, a, b