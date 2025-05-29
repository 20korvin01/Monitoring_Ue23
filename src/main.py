import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from plots import plot_neigungsdaten, plot_delta_t, plot_autokovarianz, plot_autokorrelation, plot_kreuzkovarianz, plot_kreuzkorrelation


def linear_interpolation(timestamps, values, t_new):
    """
    Interpoliert die Werte auf die neuen Zeitstempel
    t_new mithilfe der linearen Interpolation.
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


def linear_regression(t, y):
    """
    Berechnet die lineare Regression der Daten y in Abhängigkeit von t.
    """
    # Berechnung der Koeffizienten
    n = len(t)
    m = (n * np.sum(t * y) - np.sum(t) * np.sum(y)) / (n * np.sum(t**2) - np.sum(t)**2)
    b = (np.sum(y) - m * np.sum(t)) / n
    
    # Berechnung der Regressionsgeraden
    y_trend = m * t + b
    return y_trend


def autocovariance(x, lag):
    """
    Berechnet die diskrete Autokovarianz der Zeitreihe x mit dem angegebenen Lag (k)
    """
    n = len(x)
    mean_x = np.mean(x)
    cov_x = np.zeros(lag)
    for k in range(lag):
        cov_x[k] = np.sum((x[:n-k] - mean_x) * (x[k:] - mean_x)) / (n - k - 1)
    return cov_x

def autocorrelation(x, cov_x):
    """
    Berechnet die Autokorrelation der Zeitreihe x
    """
    var_x = np.var(x)
    acf = cov_x / var_x
    return acf


def crosscovariance(x, y, lag):
    """
    Berechnet die diskrete Kreuzkovarianz der Zeitreihen x und y mit dem angegebenen Lag (k)
    """
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_xy = np.zeros(lag)
    for k in range(lag):
        cov_xy[k] = np.sum((x[:n-k] - mean_x) * (y[k:] - mean_y)) / (n - k - 1)
    return cov_xy

def crosscorrelation(x, y, crosscov_xy):
    """
    Berechnet die Kreuzkorrelation der Zeitreihen x und y
    """
    var_x = np.var(x)
    var_y = np.var(y)
    crosscorr = crosscov_xy / np.sqrt(var_x * var_y)
    return crosscorr

def dft(signal):
    """
    Berechnet die diskrete Fourier-Transformation (DFT) des Signals.
    """
    n = len(signal)
    X = []

    for k in range(n):
        X_k = 0

        for i in range(n):
            e = np.exp(2j * np.pi *k * i/n)
            X_k += signal[i]/e
        X.append(X_k)
    return np.array(X)

def leistungsdichtespektrum(acf, dt):
    """
    Berechnet das Leistungsdichtespektrum (LDS) aus der Autokorrelationsfunktion (ACF).
    """
    
    m = len(acf)
    lds = np.zeros(m)

    for k in range(m):
        summe = 0
        for j in range(1, m - 1):
            summe += acf[j] * np.cos(np.pi * k * j / m)

        lds[k] = 4 * dt * (
            0.5 * acf[0] +
            0.5 * (-1)**k * acf[m - 1] +
            summe
        )
    return lds

def amplitudenspektrum(lds, dt):
    """
    Berechnet das Amplitudenspektrum aus dem Leistungsdichtespektrum (LDS).
    """
    m = len(lds)
    A_k = np.zeros(m)
    for k in range(m):
        A_k[k] = np.sqrt(lds[k] / (m * dt))
    return A_k

if __name__ == "__main__":   
    
    ### AUFGABE 0 ####################################################################################
    """Laden und Darstellen der Messwerte"""
    # Laden der Messwerte
    mat_contents = scipy.io.loadmat('data/Neigung.mat')
    # Neigung-x, Neigung-y, Temperatur, Zeit in s || Abtastinterval: 120s
    neigung_zeitreihe = np.array(mat_contents['N'])
    neigung_x = neigung_zeitreihe[:, 0]
    neigung_y = neigung_zeitreihe[:, 1]
    temperatur = neigung_zeitreihe[:, 2]
    t = neigung_zeitreihe[:, 3]
    n = len(t)
    # Plotten der Messwerte
    # plot_neigungsdaten(neigung_zeitreihe)
    
    
       
    
    ### AUFGABE 1 ####################################################################################
    """Aufbereitung der Messwerte"""
    ## 1.1 Anschauliche Darstellung der Datenlücken ----------------------------------------------- ##
    # --> dazu Berechnen der Differenzen der Zeitstempel (dort wo Differenz > 120s)
    # plot_delta_t(neigung_zeitreihe[:, 3])
    
    ## 1.2 Füllen der Datenlücken durch lineare Interpolation ------------------------------------- ##
    # Abtastintervall
    dT = 120
    # Start- und Endzeit
    t_start = t[0]
    t_end = t[-1]
    # neue Zeitreihe
    t_new = np.arange(t_start, t_end, dT)
    # lineare Interpolation der Neigungsdaten
    neigung_x_interp = linear_interpolation(t, neigung_x, t_new)
    neigung_y_interp = linear_interpolation(t, neigung_y, t_new)
    temperatur_interp = linear_interpolation(t, temperatur, t_new)
    
    ## 1.3 Beseitigung vorhandener (linearer) Trends --------------------------------------------- ##
    # Berechnung der linearen Regression (y = mx + b)
    neigung_x_trend = linear_regression(t_new, neigung_x_interp)
    neigung_y_trend = linear_regression(t_new, neigung_y_interp)
    temperatur_trend = linear_regression(t_new, temperatur_interp)
    # Subtraktion der Trends -> stationäre Zeitreihen
    neigung_x_interp_detrend = neigung_x_interp - neigung_x_trend + neigung_x_trend[0]
    neigung_y_interp_detrend = neigung_y_interp - neigung_y_trend + neigung_y_trend[0]
    temperatur_interp_detrend = temperatur_interp - temperatur_trend + temperatur_trend[0]


    ### AUFGABE 2 ###################################################################################
    """Autokovarianz"""
    ## 2.1 Berechnung der Autokovarianz- und Autokorrelationsfunktionen -------------------------- ##
    lag = n//10 - 3 # n=7080 Datapoints bei dt=120s = 10 Tage und 1h Aufnahmedauer -> 10d+1h /10 = 1d + 6min => 6min = 120s * 3 Aufnahmen
    cov_x = autocovariance(neigung_x_interp_detrend, lag)
    cov_y = autocovariance(neigung_y_interp_detrend, lag)
    cov_t = autocovariance(temperatur_interp_detrend, lag)
    acf_x = autocorrelation(neigung_x_interp_detrend, cov_x)
    acf_y = autocorrelation(neigung_y_interp_detrend, cov_y)
    acf_t = autocorrelation(temperatur_interp_detrend, cov_t)
    
    ## 2.2 Darstellung der Autokovarianz- und Autokorrelationsfunktionen ------------------------- ##
    # plot_autokovarianz(cov_x, cov_y, cov_t)
    # plot_autokorrelation(acf_x, acf_y, acf_t)
    
    ## 2.3 Interpretation der Verläufe der Autokorrelationsfunktionen ---------------------------- ##
    
    ## 2.4 Interpretation der Stellen C(0), C(1) und C(τ>1) der Autokovarianz -------------------- ##
    
    ## 2.5 Gauß-Markov-Prozess, weißes Rauschen oder farbiges Rauschen? -------------------------- ##
    
    
    ### AUFGABE 3 ###################################################################################
    """Kreuzkovarianz"""
    ## 3.1 Berechnung der drei Kombinationen der Kreuzkovarianz und -korrelationsfunktionen ------ ##
    lag = n//10 -3
    crosscov_xy = crosscovariance(neigung_x_interp_detrend, neigung_y_interp_detrend, lag)
    crosscov_xt = crosscovariance(neigung_x_interp_detrend, temperatur_interp_detrend, lag)
    crosscov_yt = crosscovariance(neigung_y_interp_detrend, temperatur_interp_detrend, lag)
    crosscorr_xy = crosscorrelation(neigung_x_interp_detrend, neigung_y_interp_detrend, crosscov_xy)
    crosscorr_xt = crosscorrelation(neigung_x_interp_detrend, temperatur_interp_detrend, crosscov_xt)
    crosscorr_yt = crosscorrelation(neigung_y_interp_detrend, temperatur_interp_detrend, crosscov_yt)
    
    ## 3.2 Darstellung der Kreuzkovarianz- und -korrelationsfunktionen --------------------------- ##
    # plot_kreuzkovarianz(crosscov_xy, crosscov_xt, crosscov_yt)
    # plot_kreuzkorrelation(crosscorr_xy, crosscorr_xt, crosscorr_yt)
    
    ## 3.3 Interpretation der Verläufe der Kreuzkorrelationsfunktionen --------------------------- ##
    
    
    ## AUFGABE 4 ####################################################################################
    """Spektralanalyse"""
    ## 4.1 Berechnung der Leistungs- und Amplitudenspektren -------------------------------------- ##
    # Daten berechnen
    m = len(acf_y)  # Anzahl der Lags
    lds = leistungsdichtespektrum(acf_y, dT)
    lds = abs(lds)  # Absolutwert, um negative Werte zu vermeiden
    np.savetxt('data/leistungsdichtespektrum.txt', lds)
    
    amp = amplitudenspektrum(lds, dT)
    # Frequenzachse
    freqs = np.arange(m) / (2 * m * dT)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, lds)
    plt.title("Leistungsdichtespektrumsignal")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD")
    plt.xlim(0, 1e-4)


    plt.subplot(2, 1, 2)
    plt.plot(freqs, amp)
    plt.title("Amplitudenspektrumsignal")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-4)

    plt.tight_layout()
    plt.show()

    ## 4.2 Interpretation der Spektren ----------------------------------------------------------- ##
    
    ## 4.3 Vergleich eigener Ergebnisse mit Ergebnissen der fft-Funktion aus dem Modul scipy ----- ##
