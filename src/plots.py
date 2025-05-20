import numpy as np
import matplotlib.pyplot as plt


def plot_neigungsdaten(neigung_zeitreihe):
    """
    Plottet die x-, y- und Temperaturdaten der Neigung
    in drei separaten Plots neben einander.
    """
    
    plt.figure(figsize=(12, 4))
    
    # Neigung x
    plt.subplot(1, 3, 1)
    plt.plot(neigung_zeitreihe[:, 0], 'b')
    plt.setp(plt.gca().lines, linewidth=0.5)
    plt.title('Neigung x')
    plt.xlabel('t [s]')
    plt.ylabel('Neigung x-Richtung [°]')

    # Neigung y
    plt.subplot(1, 3, 2)
    plt.plot(neigung_zeitreihe[:, 1], 'b')
    plt.setp(plt.gca().lines, linewidth=0.5)
    plt.title('Neigung y')
    plt.xlabel('t [s]')
    plt.ylabel('Neigung y-Richtung [°]')

    # Temperatur
    plt.subplot(1, 3, 3)
    plt.plot(neigung_zeitreihe[:, 2], 'b')
    plt.setp(plt.gca().lines, linewidth=0.5)
    plt.title('Temperatur')
    plt.xlabel('t [s]')
    plt.ylabel('Temperatur [°C]')

    plt.tight_layout()
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
    plt.xlabel('t [s]')
    plt.ylabel('Δt [s]')
    plt.legend()
    plt.tight_layout()
    plt.show()