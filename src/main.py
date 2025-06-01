import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy.fft
import sys

from plots import plot_neigungsdaten, plot_delta_t, plot_autokovarianz, plot_autokorrelation, plot_kreuzkovarianz, plot_kreuzkorrelation, plot_leistungsdichtespektrum
from plots import plot_neigungsdaten, plot_delta_t, plot_autokovarianz, plot_autokorrelation, plot_kreuzkovarianz, plot_kreuzkorrelation


def linear_interpolation(timestamps, values, t_new):
    """
    Interpoliert die Werte auf die neuen Zeitstempel
    t_new mithilfe der linearen Interpolation.
    """
    values_interp = np.zeros(len(t_new))
    is_interp = np.zeros(len(t_new))
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
            is_interp[i] = 1  # Markiere, dass dieser Wert interpoliert wurde
            
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

        lds[k] = 2 * dt * (
            acf[0] +
            (-1)**k * acf[m - 1] +
            2*summe
        ) # statt 4*(0.5*acf +0.5*acf +summe) jetzt 2*(acf +acf +2*summe) => spart eine Multiplikation, wenn schon keine inneren Klammern verwendet werden
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
    mat_contents = scipy.io.loadmat('../data/Neigung.mat')
    # Neigung-x, Neigung-y, Temperatur, Zeit in s || Abtastinterval: 120s
    neigung_zeitreihe = np.array(mat_contents['N'])
    #print(neigung_zeitreihe.shape)
    neigung_x = neigung_zeitreihe[:, 0]
    neigung_y = neigung_zeitreihe[:, 1]
    temperatur = neigung_zeitreihe[:, 2]
    t = neigung_zeitreihe[:, 3]
    n = len(t)
    print(f"Anzahl Messungen: {n}")
    # Plotten der Messwerte
    # plot_neigungsdaten(neigung_zeitreihe)




    #plot_neigungsdaten(neigung_zeitreihe)
       
    
    ### AUFGABE 1 ####################################################################################
    """Aufbereitung der Messwerte"""
    ## 1.1 Anschauliche Darstellung der Datenlücken ----------------------------------------------- ##
    # --> dazu Berechnen der Differenzen der Zeitstempel (dort wo Differenz > 120s)
    # plot_delta_t(neigung_zeitreihe[:, 3])

    plot_delta_t(neigung_zeitreihe[:, 3])
    t_soll = np.arange(t[0], t[-1], 120)  # Soll-Zeitreihe
    print(f"Anzahl der Soll-Zeitstempel: {len(t_soll)}")
    luecken = []
    neigung_x_ol = []
    neigung_y_ol = []
    temperatur_ol = []
    t_ol = []
    neigung_x_ft = []
    neigung_y_ft = []
    temperatur_ft = []
    t_ft = []
    for i in t_soll:
        if i not in t:
            #print(f"Zeitstempel {i} fehlt in den Messdaten.")
            luecken.append(i)
        else:
            neigung_x_ol.append(neigung_x[t == i][0])
            neigung_y_ol.append(neigung_y[t == i][0])
            temperatur_ol.append(temperatur[t == i][0])
            t_ol.append(i)
        if t[(t>i)&(t<i+120)].size > 0:
            neigung_x_ft.append(neigung_x[t == t[(t>i) & (t<i+120)]])
            neigung_y_ft.append(neigung_y[t == t[(t>i) & (t<i+120)][0]][0])
            temperatur_ft.append(temperatur[t == t[(t>i) & (t<i+120)]])
            t_ft.append(t[(t>i) & (t<i+120)])
    print(f"Anzahl der Messungen zu Zeitpunkten k*120s: {len(t_ol)}")
    print(f"Anzahl der fehlenden Messungen zu Zeitpunkten k*120s: {len(luecken)}")
    print(f"Anzahl der Messungen zu Zeitpunkt != k*120s: {len(t_ft)}")
    #[array([[ -0.27  ,  -0.6804,  26.02  , 120.    ]]), array([[ -0.27  ,  -0.6805,  26.02  , 240.    ]]), array([[-2.700e-01, -6.806e-01,  2.602e+01,  3.600e+02]]), array([[-2.700e-01, -6.806e-01,  2.602e+01,  4.800e+02]]), array([[-2.701e-01, -6.804e-01,  2.603e+01,  6.000e+02]])]

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.multiply(t_ol, 1/(60*60*24)), neigung_x_ol, 'xg', markersize=2, label='Korrekte Messung, t = k*120s')
    plt.plot(np.multiply(t_ft, 1/(60*60*24)), neigung_x_ft, '.b', markersize=1, label='Falscher Zeitpunkt, t != k*120s')
    plt.vlines(np.multiply(luecken, 1/(60*60*24)), ymin=np.mean(neigung_x_ol)-0.0005, ymax=np.mean(neigung_x_ol)+0.0005, color='r', linewidth=0.1, label='Fehlende Messung')
    plt.ylabel('Neigung in x-Richtung in °')
    plt.xlim(t[0]/(60*60*24), t[-1]/(60*60*24))
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.4))

    plt.subplot(3, 1, 2)
    plt.plot(np.multiply(t_ol, 1/(60*60*24)), neigung_y_ol, 'xg', markersize=2, label='Korrekte Messung, t = k*120s')
    plt.plot(np.multiply(t_ft, 1/(60*60*24)), neigung_y_ft, '.b', markersize=1, label='Falscher Zeitpunkt, t != k*120s')
    plt.vlines(np.multiply(luecken, 1/(60*60*24)), ymin=np.mean(neigung_y_ol)-0.0003, ymax=np.mean(neigung_y_ol)+0.0003, color='r', linewidth=0.1, label='Fehlende Messung')
    plt.ylabel('Neigung in y-Richtung in °')
    plt.xlim(t[0]/(60*60*24), t[-1]/(60*60*24))

    plt.subplot(3, 1, 3)
    plt.plot(np.multiply(t_ol, 1/(60*60*24)), temperatur_ol, 'xg', markersize=2, label='Korrekte Messung, t = k*120s')
    plt.plot(np.multiply(t_ft, 1/(60*60*24)), temperatur_ft, '.b', markersize=1, label='Falscher Zeitpunkt, t != k*120s')
    plt.vlines(np.multiply(luecken, 1/(60*60*24)), ymin=np.mean(temperatur_ol)-0.07, ymax=np.mean(temperatur_ol)+0.07, color='r', linewidth=0.1, label='Fehlende Messung')
    plt.xlabel('t in Tagen')
    plt.ylabel('Temperatur in °C')
    plt.xlim(t[0]/(60*60*24), t[-1]/(60*60*24))

    plt.tight_layout()
    plt.suptitle('Messdaten mit Lücken', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig('../plots/neigung_x_zeitreihe_luecken.png')
    #plt.show()

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


    # plt.figure(figsize=(12, 4))
    # # Neigung x
    # plt.subplot(1, 3, 1)
    # plt.plot(neigung_x_interp, 'b')
    # plt.setp(plt.gca().lines, linewidth=0.5)
    # plt.title('Neigung x interpoliert')
    # plt.xlabel('t [s]')
    # plt.ylabel('Neigung x-Richtung [°]')

    # # Neigung y
    # plt.subplot(1, 3, 2)
    # plt.plot(neigung_y_interp, 'b')
    # plt.setp(plt.gca().lines, linewidth=0.5)
    # plt.title('Neigung y interpoliert')
    # plt.xlabel('t [s]')
    # plt.ylabel('Neigung y-Richtung [°]')

    # # Temperatur
    # plt.subplot(1, 3, 3)
    # plt.plot(temperatur_interp, 'b')
    # plt.setp(plt.gca().lines, linewidth=0.5)
    # plt.title('Temperatur interpoliert')
    # plt.xlabel('t [s]')
    # plt.ylabel('Temperatur [°C]')

    # plt.tight_layout()
    # plt.savefig('plots/neigung_zeitreihe_interp.png')
    # plt.show()
    
    ## 1.3 Beseitigung vorhandener (linearer) Trends --------------------------------------------- ##
    # Berechnung der linearen Regression (y = mx + b)
    neigung_x_trend = linear_regression(t_new, neigung_x_interp)
    neigung_y_trend = linear_regression(t_new, neigung_y_interp)
    temperatur_trend = linear_regression(t_new, temperatur_interp)
    # Subtraktion der Trends -> stationäre Zeitreihen
    neigung_x_interp_detrend = neigung_x_interp - neigung_x_trend + neigung_x_trend[0]
    neigung_y_interp_detrend = neigung_y_interp - neigung_y_trend + neigung_y_trend[0]
    temperatur_interp_detrend = temperatur_interp - temperatur_trend + temperatur_trend[0]



    # Interpolierte Daten
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), neigung_x_interp, '.g', markersize=1)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), neigung_x_trend, 'r', linewidth=0.5)
    plt.ylabel('Neigung in x-Richtung in °')
    plt.xlim(t_new[0]/(60*60*24), t_new[-1]/(60*60*24))

    plt.subplot(3, 1, 2)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), neigung_y_interp, '.g', markersize=1)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), neigung_y_trend, 'r', linewidth=0.5)
    plt.ylabel('Neigung in y-Richtung in °')
    plt.xlim(t_new[0]/(60*60*24), t_new[-1]/(60*60*24))

    plt.subplot(3, 1, 3)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), temperatur_interp, '.g', markersize=1)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), temperatur_trend, 'r', linewidth=0.5)
    plt.xlabel('t in Tagen')
    plt.ylabel('Temperatur in °C')
    plt.xlim(t_new[0]/(60*60*24), t_new[-1]/(60*60*24))

    plt.tight_layout()
    plt.suptitle('Interpolierte Messdaten', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig('../plots/interpoliert.png')

    # Trendbereinigte Daten
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), neigung_x_interp_detrend, '.g', markersize=1)
    plt.ylabel('Neigung in x-Richtung in °')
    plt.xlim(t_new[0]/(60*60*24), t_new[-1]/(60*60*24))

    plt.subplot(3, 1, 2)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), neigung_y_interp_detrend, '.g', markersize=1)
    plt.ylabel('Neigung in y-Richtung in °')
    plt.xlim(t_new[0]/(60*60*24), t_new[-1]/(60*60*24))

    plt.subplot(3, 1, 3)
    plt.plot(np.multiply(t_new, 1/(60*60*24)), temperatur_interp_detrend, '.g', markersize=1)
    plt.xlabel('t in Tagen')
    plt.ylabel('Temperatur in °C')
    plt.xlim(t_new[0]/(60*60*24), t_new[-1]/(60*60*24))

    plt.tight_layout()
    plt.suptitle('Interpolierte und trendbereinigte Messdaten', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig('../plots/detrend.png')

    # plt.figure(figsize=(12, 4))
    # # Neigung x
    # plt.subplot(1, 3, 1)
    # plt.plot(neigung_x_interp_detrend, 'b')
    # plt.setp(plt.gca().lines, linewidth=0.5)
    # plt.title('Neigung x interpoliert ohne Trend')
    # plt.xlabel('t [s]')
    # plt.ylabel('Neigung x-Richtung [°]')

    # # Neigung y
    # plt.subplot(1, 3, 2)
    # plt.plot(neigung_y_interp_detrend, 'b')
    # plt.setp(plt.gca().lines, linewidth=0.5)
    # plt.title('Neigung y interpoliert ohne Trend')
    # plt.xlabel('t [s]')
    # plt.ylabel('Neigung y-Richtung [°]')

    # # Temperatur
    # plt.subplot(1, 3, 3)
    # plt.plot(temperatur_interp_detrend, 'b')
    # plt.setp(plt.gca().lines, linewidth=0.5)
    # plt.title('Temperatur interpoliert ohne Trend')
    # plt.xlabel('t [s]')
    # plt.ylabel('Temperatur [°C]')

    # plt.tight_layout()
    # plt.savefig('plots/neigung_zeitreihe_interp_detrend.png')
    # plt.show()
    # sys.exit(0)  # Stop execution after task 1
    
    ### AUFGABE 2 ###################################################################################
    """Autokovarianz"""
    ## 2.1 Berechnung der Autokovarianz- und Autokorrelationsfunktionen -------------------------- ##
    lag = n//10 +12# n=7080 Datapoints bei dt=120s = 9 Tage und 20h Aufnahmedauer -> 9d+20h /10 = 23h 36min => 24min = 120s * 12 Aufnahmen, welche pro Tag fehlen
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
    lag = n//10+12
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
    lds_y = abs(dft(acf_y))# Absolutwert, um negative Werte zu vermeiden
    lds_x = abs(dft(acf_x))
    np.savetxt('../data/leistungsdichtespektrum_x.txt', lds_x)
    np.savetxt('../data/leistungsdichtespektrum_y.txt', lds_y)

    amp_x = amplitudenspektrum(lds_x, dT)
    amp_y= amplitudenspektrum(lds_y, dT)
    # Frequenzachse
    freqs = np.arange(m) / (m * dT)
    fn = 1/(2*dT)
    # f_lim = freqs[1]+fn/2
    f_lim = freqs[1]+fn
    plot_leistungsdichtespektrum(freqs,lds_x,lds_y,amp_x,amp_y)


    ## 4.2 Interpretation der Spektren ----------------------------------------------------------- ##

    ## 4.3 Vergleich eigener Ergebnisse mit Ergebnissen der fft-Funktion aus dem Modul scipy ----- ##

    lds_x_scipy = abs(scipy.fft.fft(acf_x))
    lds_y_scipy = abs(scipy.fft.fft(acf_y))

    np.savetxt('../data/leistungsdichtespektrum_scipy_x.txt', lds_x_scipy)
    np.savetxt('../data/leistungsdichtespektrum_scipy_y.txt', lds_y_scipy)

    amp_x_scipy = amplitudenspektrum(lds_x_scipy, dT)
    amp_y_scipy = amplitudenspektrum(lds_y_scipy, dT)

    # nicht separat, da anderer Dateiname beim Export gebraucht wird
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, lds_x_scipy, 'r', label='x-Richtung')
    plt.plot(freqs, lds_y_scipy, 'b', label='y-Richtung')
    # plt.title("Leistungsdichtespektrum")
    plt.xlabel("Frequenz in Hz")
    plt.ylabel("LDS in Grad²/Hz")
    plt.legend()
    plt.xlim(0, f_lim) # nur bis zur halben Nyquistfrequenz plotten, da es sich um ein reelles Signal handelt und nur m/2 Werte unabh. sind

    plt.subplot(2, 1, 2)
    plt.semilogy(freqs, amp_x_scipy, 'r', label='x-Richtung')
    plt.semilogy(freqs, amp_y_scipy, 'b', label='y-Richtung')
    # plt.title("Amplitudenspektrum")
    plt.xlabel("Frequenz in Hz")
    plt.ylabel("Amplitude in Grad")
    plt.legend()
    plt.xlim(0, f_lim)

    plt.tight_layout()
    plt.savefig('../plots/leistungsdichtespektrum_scipy.png')


    # Plot zoomed
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.stem(freqs, lds_x_scipy, 'r', basefmt=" ", label='x-Richtung')
    plt.stem(freqs, lds_y_scipy, 'b', basefmt=" ", label='y-Richtung')
    # plt.title("Leistungsdichtespektrum")
    plt.xlabel("Frequenz in µHz")
    plt.ylabel("LDS in Grad²/Hz")
    plt.legend()
    plt.xlim(0, 1e-4)
    plt.ylim(bottom=0)
    plt.xticks(freqs[:9],[f"{val:.3f}" for val in freqs[:9]*1e6])

    plt.subplot(2, 1, 2)
    plt.stem(freqs, amp_x_scipy, 'r', basefmt=" ", label='x-Richtung')
    plt.stem(freqs, amp_y_scipy, 'b', basefmt=" ", label='y-Richtung')
    # plt.title("Amplitudenspektrum")
    # plt.yscale('log')
    plt.xlabel("Frequenz in µHz")
    plt.ylabel("Amplitude in Grad")
    plt.legend()
    plt.xlim(0, 1e-4)
    plt.ylim(bottom=0)
    plt.xticks(freqs[:9],[f"{val:.3f}" for val in freqs[:9]*1e6])

    plt.tight_layout()
    plt.savefig('../plots/LDS_zoomed_scipy.png')
    plt.show()