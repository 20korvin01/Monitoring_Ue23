import numpy as np
import matplotlib.pyplot as plt


def plot_neigungsdaten(neigung_zeitreihe):
    """
    Plottet die x-, y- und Temperaturdaten der Neigung
    in drei separaten Plots neben einander.
    """
    # / 60s*60min # / 24h
    # Zeit in Tagen
    time = neigung_zeitreihe[:,3]/3600/24

    plt.figure(figsize=(12, 4))
    
    # Neigung x
    plt.subplot(1, 3, 1)
    plt.plot(time,neigung_zeitreihe[:, 0], 'b')
    plt.setp(plt.gca().lines, linewidth=0.5)
    plt.title('Neigung x')
    plt.xlabel('t in Tagen')
    plt.ylabel('Neigung x-Richtung in °')

    # Neigung y
    plt.subplot(1, 3, 2)
    plt.plot(time,neigung_zeitreihe[:, 1], 'b')
    plt.setp(plt.gca().lines, linewidth=0.5)
    plt.title('Neigung y')
    plt.xlabel('t in Tagen')
    plt.ylabel('Neigung y-Richtung in °')

    # Temperatur
    plt.subplot(1, 3, 3)
    plt.plot(time,neigung_zeitreihe[:, 2], 'b')
    plt.setp(plt.gca().lines, linewidth=0.5)
    plt.title('Temperatur')
    plt.xlabel('t in Tagen')
    plt.ylabel('Temperatur in °C')

    plt.tight_layout()
    plt.savefig('plots/neigung_zeitreihe.png')
    plt.show()



def plot_delta_t(timestamps):
    """
    Plottet die Differenzen der Zeitstempel.
    In blau: delta_t = 120s
    In rot: delta_t > 120s
    In grün: delta_t < 120s
    """
    delta_t = np.diff(timestamps)

    blau = np.array([timestamps[:-1][delta_t == 120], delta_t[delta_t == 120]]).T
    rot = np.array([timestamps[:-1][delta_t > 120], delta_t[delta_t > 120]]).T
    gruen = np.array([timestamps[:-1][delta_t < 120], delta_t[delta_t < 120]]).T

    # plot
    plt.figure(figsize=(12, 4))
    plt.plot(blau[:, 0], blau[:, 1], 'b', label='Δt = 120s', marker='o', markersize=2, ls="")
    plt.plot(rot[:, 0], rot[:, 1], 'r', label='Δt > 120s', marker='o', markersize=2, ls="")
    plt.plot(gruen[:, 0], gruen[:, 1], 'g', label='Δt < 120s', marker='o', markersize=2, ls="")

    plt.title('Differenzen der Zeitstempel')
    plt.xlabel('t in s')
    plt.ylabel('Δt in s')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/delta_t.png')
    plt.show()



def plot_autokovarianz(cov_x, cov_y, cov_t):
    """
    Plottet die Autokovarianzen der Neigung x, y und Temperatur
    in drei separaten Plots neben einander.
    """
    time = np.arange(len(cov_x)) * 120 / 3600

    plt.figure(figsize=(12, 4))

    # Autokovarianz Neigung x
    plt.subplot(1, 3, 1)
    plt.plot(time, cov_x, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Autokovarianz Neigung x')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Autokovarianz')

    # Autokovarianz Neigung y
    plt.subplot(1, 3, 2)
    plt.plot(time, cov_y, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Autokovarianz Neigung y')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Autokovarianz')

    # Autokovarianz Temperatur
    plt.subplot(1, 3, 3)
    plt.plot(time, cov_t, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Autokovarianz Temperatur')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Autokovarianz')

    plt.tight_layout()
    plt.savefig('plots/autokovarianz_zeitreihe.png')
    plt.show()



def plot_autokorrelation(acf_x, acf_y, acf_t):
    """
    Plottet die Autokorrelationsfunktionen der Neigung x, y und Temperatur
    in drei separaten Plots neben einander.
    """
    time = np.arange(len(acf_x)) * 120 / 3600

    plt.figure(figsize=(12, 4))

    # Autokorrelation Neigung x
    plt.subplot(1, 3, 1)
    plt.plot(time, acf_x, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Autokorrelation Neigung x')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Autokorrelationskoeffizient')

    # Autokorrelation Neigung y
    plt.subplot(1, 3, 2)
    plt.plot(time, acf_y, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Autokorrelation Neigung y')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Autokorrelationskoeffizient')

    # Autokorrelation Temperatur
    plt.subplot(1, 3, 3)
    plt.plot(time, acf_t, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Autokorrelation Temperatur')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Autokorrelationskoeffizient')

    plt.tight_layout()
    plt.savefig('plots/autokorrelation_zeitreihe.png')
    plt.show()



def plot_kreuzkovarianz(crosscov_xy, crosscov_xt, crosscov_yt):
    """
    Plottet die Kreuzkovarianzen der Neigung x, y und Temperatur
    in drei separaten Plots neben einander.
    """
    time = np.arange(len(crosscov_xy)) * 120 / 3600

    plt.figure(figsize=(12, 4))

    # Kreuzkovarianz Neigung x und y
    plt.subplot(1, 3, 1)
    plt.plot(time, crosscov_xy, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Kreuzkovarianz Neigung x und y')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Kreuzkovarianz')

    # Kreuzkovarianz Neigung x und Temperatur
    plt.subplot(1, 3, 2)
    plt.plot(time, crosscov_xt, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Kreuzkovarianz Neigung x und Temperatur')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Kreuzkovarianz')

    # Kreuzkovarianz Neigung y und Temperatur
    plt.subplot(1, 3, 3)
    plt.plot(time, crosscov_yt, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Kreuzkovarianz Neigung y und Temperatur')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Kreuzkovarianz')

    plt.tight_layout()
    plt.savefig('plots/kreuzkovarianz_zeitreihe.png')
    plt.show()



def plot_kreuzkorrelation(crosscorr_xy, crosscorr_xt, crosscorr_yt):
    """
    Plottet die Kreuzkorrelationsfunktionen der Neigung x, y und Temperatur
    in drei separaten Plots neben einander.
    """
    time = np.arange(len(crosscorr_xy)) * 120 / 3600

    plt.figure(figsize=(12, 4))

    # Kreuzkorrelation Neigung x und y
    plt.subplot(1, 3, 1)
    plt.plot(time, crosscorr_xy, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Kreuzkorrelation Neigung x und y')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Kreuzkorrelationskoeffizient')

    # Kreuzkorrelation Neigung x und Temperatur
    plt.subplot(1, 3, 2)
    plt.plot(time, crosscorr_xt, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Kreuzkorrelation Neigung x und Temperatur')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Kreuzkorrelationskoeffizient')

    # Kreuzkorrelation Neigung y und Temperatur
    plt.subplot(1, 3, 3)
    plt.plot(time, crosscorr_yt, 'b', marker='o', markersize=0.5, ls="")
    plt.title('Kreuzkorrelation Neigung y und Temperatur')
    plt.xlabel('Zeitabstand τ in h')
    plt.ylabel('Kreuzkorrelationskoeffizient')

    plt.tight_layout()
    plt.savefig('plots/kreuzkorrelation_zeitreihe.png')
    plt.show()

def plot_leistungsdichtespektrum(freqs,lds_x,lds_y,amp_x,amp_y):
    # Plot
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, lds_x,'r',label='x-Richtung')
    plt.plot(freqs, lds_y,'b',label='y-Richtung')
    plt.title("Leistungsdichtespektrum")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD")
    plt.legend()
    plt.xlim(0, 1e-4)

    plt.subplot(2, 1, 2)
    plt.semilogy(freqs, amp_x,'r',label='x-Richtung')
    plt.semilogy(freqs, amp_y,'b',label='y-Richtung')
    plt.title("Amplitudenspektrum")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim(0, 1e-4)


    plt.tight_layout()
    plt.savefig('plots/leistungsdichtespektrum.png')
    plt.show()