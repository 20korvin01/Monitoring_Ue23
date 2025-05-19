# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import xcorr as corr
import perform_fft as perform
from scipy import signal

def Trend_Residuen(zeit, reihe):
    trend = np.polyfit(zeit, reihe,3)
    pol = np.polyval(trend, zeit)
    residuen = reihe-pol
    
    return pol, residuen

def Interpolation(reihe):
    reihe_umdefiniert = np.arange(reihe.shape[0])
    Werte_NaN = np.where(np.isfinite(reihe)) #Prüfung welche Werte NaN sind
    interp_miss1 = interpolate.interp1d(reihe_umdefiniert[Werte_NaN], reihe[Werte_NaN],bounds_error=False, kind = 'cubic')

    reihe_interpoliert = np.where(np.isfinite(reihe),reihe,interp_miss1(reihe_umdefiniert))
    
    return reihe_interpoliert

def Ausreißer_gleich_NaN(residuen, Sigma_pos, Sigma_neg):
    residuen_ohne_Ausreißer = residuen
    Anzahl = 0
    for i in range(len(residuen)):
        if residuen[i] > Sigma_pos[0] or residuen[i] < Sigma_neg[0]:
            residuen_ohne_Ausreißer = np.delete(residuen_ohne_Ausreißer,i)
            residuen_ohne_Ausreißer = np.insert(residuen_ohne_Ausreißer,i,np.NaN)
            Anzahl += 1
    
    return residuen_ohne_Ausreißer, Anzahl

def Glaettung(eingabearray1, eingabearray2):
    '''Gibt die diskrete, lineare Faltung zweier eindimensionaler Folgen zurück'''
    geglaettete_Funktion = np.convolve(eingabearray1, eingabearray2, mode='same')
    return geglaettete_Funktion
    

if __name__=='__main__':
    with open ('time_series.txt') as file:
        daten = file.readlines()
    file.close()

    
    date_time = []
    date = []
    time = []
    reihe1 = []
    reihe2 = []
    
    for i in daten:
        date_time.append(str(i[0:19]))
        #date.append(i[0:10])
        #time.append(i[11:20])
        daten=i.split(" ")
        reihe1.append(float(daten[2]))
        #reihe1.append(float(i[20:26]))
        reihe2.append(float(daten[3]))

        
    reihe1 = np.array(reihe1)
    reihe2 = np.array(reihe2)
        
    '''Zeit als datetime und als Zeitstempel formatieren'''
    datum_zeit = [datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S").replace() for date in date_time]
    
    zeitstempel = [datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S").timestamp() for date in date_time]

    
    '''Berechnung des Trends und der Residuen'''
    #Reihe 1
    #Interpolation der 5 fehlenden (NaN) Werte in der Zeitreihe
    reihe1_1 = Interpolation(reihe1)
    pol1_1, residuen1_1 = Trend_Residuen(zeitstempel, reihe1_1)

    #Reihe 2
    reihe2_1 = reihe2
    pol2_1, residuen2_1 = Trend_Residuen(zeitstempel, reihe2_1)

    '''Datenlücken detektieren'''
    delta_t = zeitstempel[1]-zeitstempel[0]
    
    # In Reihe 1 gibt es zusätzlich 5 NaN Werte, die schon vorher vorhanden waren
    #Prüfung wo diese snd:
# =============================================================================
#     for i in range(len(reihe1)):
#         if np.isnan(reihe1[i]):
#             print(zeitstempel[i])
# =============================================================================
            
    # Fehlenden Zeitwerte in reihe 1 und 2
    # Überall wo delta_t nicht 10s ist wird np:NaN in die Zeitreihen eingefügt
    # und der jeweilige Zeitstempel wird zum Array der Zeitstempel hinzugefügt    
    zeitstempel_arr = np.array(zeitstempel)  #Umwandlung der Zeitstempel in Array
    for i in range(len(reihe1)):
        if zeitstempel[i] - zeitstempel[i-1] != delta_t:            
            reihe1 = np.insert(reihe1,i-1,np.NaN)
            reihe2 = np.insert(reihe2,i-1, np.NaN)
            zeitstempel_arr = np.insert(zeitstempel_arr, (i-1), [zeitstempel[i-1]+delta_t])
    
# =============================================================================
#     #Prüfung ob NaN in Reihe
#     for i in range(len(reihe2)):
#         if np.isnan(reihe2[i]):
#             print(reihe2[i])
# =============================================================================
    
    '''Zeitstempel als liste und als neue lange datetime-Liste formatieren'''
    zeitstempel = zeitstempel_arr.tolist()
    
    datum_zeit_neu = [datetime.datetime.fromtimestamp(timestamp) for timestamp in zeitstempel]
    
    '''Interpolation der fehlenden Zeitwerte'''
    reihe1_interpoliert = Interpolation(reihe1)
    reihe2_interpoliert = Interpolation(reihe2)
    
    '''Erneute Berechnung des Trends und der Residuen nach der Interpolation'''
    #Reihe 1
    pol1_2, residuen1_2 = Trend_Residuen(zeitstempel, reihe1_interpoliert)

    #Reihe 2
    pol2_2, residuen2_2 = Trend_Residuen(zeitstempel, reihe2_interpoliert)
    
    '''Detektion von Ausreißern'''
    #Empirische Standardabweichung berechnen
    stabw_residuen1 = np.std(residuen1_2)
    stabw_residuen2 = np.std(residuen2_2)
    
    Sigma_1_pos = np.empty(len(reihe1))
    Sigma_1_pos.fill(stabw_residuen1 * 3)
    Sigma_1_neg = np.empty(len(reihe1))
    Sigma_1_neg.fill(stabw_residuen1 * -3)
    
    Sigma_2_pos = np.empty(len(reihe2))
    Sigma_2_pos.fill(stabw_residuen2 * 3)
    Sigma_2_neg = np.empty(len(reihe2))
    Sigma_2_neg.fill(stabw_residuen2 * -3)
    
    #Ausreißer = NaN setzen
    residuen_ohne_Ausreißer1, Anzahl_Ausreißer1 = Ausreißer_gleich_NaN(residuen1_2, Sigma_1_pos, Sigma_1_neg)
    residuen_ohne_Ausreißer2, Anzahl_Ausreißer2 = Ausreißer_gleich_NaN(residuen2_2, Sigma_2_pos, Sigma_2_neg)
    
    #Interpolation der entstandenen Lücken
    residuen_ohne_Ausreißer_interp1 = Interpolation(residuen_ohne_Ausreißer1)
    residuen_ohne_Ausreißer_interp2 = Interpolation(residuen_ohne_Ausreißer2)
    
    '''Filter: gleitendes Mittel/Tiefpassfilter'''
    array2 = np.ones(500)/500
    filter1 = Glaettung(residuen1_2, array2)
    filter2 = Glaettung(residuen2_2, array2)
    
    '''Analyse im Frequenzraum/ FFT'''
    # 1/delta_t = Erfassungsfrequenz
    frequenz1, amplitude1 = perform.perform_fft(residuen_ohne_Ausreißer_interp1, 1/delta_t)
    frequenz2, amplitude2 = perform.perform_fft(residuen_ohne_Ausreißer_interp2, 1/delta_t)
    
    #Peaks reihe 1
    peaks1, properties1 = signal.find_peaks(amplitude1, height = 2, threshold=0)
    amplitude_peak_1_1 = properties1['peak_heights'][0]
    frequenz_peak1_1 = frequenz1[peaks1[0]]

    amplitude_peak_1_2 = properties1['peak_heights'][1]
    frequenz_peak1_2 = frequenz1[peaks1[1]]
    
    #Frequenz:
    #Frequenz1 = 1.1023e-05 [Hz]
    #Frequenz2 = 2.2046e-05 [Hz]
    
    #Frequenz in Periodenlänge:
    #Periodenlänge1 = 1/ Frequenz1 / 3600 = 25.199834689084444 Stunden
    #Periodenlänge2 = 1/ Frequenz2 / 3600 = 12.599917344542222 Stunden
    
    #Peaks reihe 2
    peaks2, properties2 = signal.find_peaks(amplitude2, height = 2, threshold=0)
    amplitude_peak_2_1 = properties2['peak_heights'][0]
    frequenz_peak2_1 = frequenz2[peaks2[0]]

    amplitude_peak_2_2 = properties2['peak_heights'][1]
    frequenz_peak2_2 = frequenz2[peaks2[1]]

    ''' Auto-/Kreuzkorrelationsanalyse'''  
    autokorr1 = corr.xcorr(residuen_ohne_Ausreißer_interp1, residuen_ohne_Ausreißer_interp1)
    autokorr2 = corr.xcorr(residuen_ohne_Ausreißer_interp2, residuen_ohne_Ausreißer_interp2)
    kreuzkorr12 = corr.xcorr(residuen_ohne_Ausreißer_interp1, residuen_ohne_Ausreißer_interp2)
    
    #Finden des höchsten Korrelationswertes
    max_korr_y = np.amax(kreuzkorr12[1])
    for i in range(len(kreuzkorr12[1])):
        if kreuzkorr12[1][i] == np.amax(kreuzkorr12[1]):
            max_korr_x = kreuzkorr12[0][i]
    
    autokorr1_2 = corr.xcorr(filter1, filter1)
    autokorr2_2 = corr.xcorr(filter2, filter2)
    kreuzkorr12_2 = corr.xcorr(filter1, filter2)
    
    #Finden des höchsten Korrelationswertes
    max_korr_y_2 = np.amax(kreuzkorr12_2[1])
    for i in range(len(kreuzkorr12_2[1])):
        if kreuzkorr12_2[1][i] == np.amax(kreuzkorr12_2[1]):
            max_korr_x_2 = kreuzkorr12_2[0][i]


    '''Daten plotten'''
    #Vor der Interpolation
# =============================================================================
#     #Reihe1 kleine Zeitreihe
#     plt.figure (figsize=(18,10))
#     plt.plot (datum_zeit, reihe1_1)
#     plt.plot(datum_zeit,pol1_1)
#     plt.plot(datum_zeit,residuen1_1)
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Zeitreihe 1")
#     plt.legend () 
#     
#     #Reihe 2 kleine Zeitreihe
#     #Reihe2 mit Trend
#     plt.figure (figsize=(18,10))
#     plt.plot (datum_zeit, reihe2_1)
#     plt.plot(datum_zeit, pol2_1)
#     plt.plot(datum_zeit,residuen2_1)
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Zeitreihe 2")
#     plt.legend () 
# =============================================================================
    
# =============================================================================
#     #Reihe1 und 2 Gegenüberstellung kleine Zeitreihe
#     plt.figure (figsize=(18,10))
#     plt.plot (datum_zeit, reihe1_1, label = "Zeitreihe 1")
#     plt.plot(datum_zeit,reihe2_1, label = "Zeitreihe 2")
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Gegenüberstellung der Zeitreihen 1 und 2")
#     plt.legend () 
# =============================================================================

    #Nach der Interpolation
# =============================================================================
#     #Reihe1 interpolierte Zeitreihe
#     plt.figure (figsize=(18,10))
#     #plt.plot (datum_zeit_neu, reihe1)
#     #plt.plot(datum_zeit,pol1)
#     plt.plot(datum_zeit_neu,residuen1_2)
#     plt.plot(datum_zeit_neu, reihe1_interpoliert)
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Detektion von Ausreißern Zeitreihe 1")
#     plt.legend () 
# =============================================================================
    
# =============================================================================
#     #Detektion von Ausreißern Reihe 1
#     plt.figure (figsize=(18,10))
#     plt.plot(datum_zeit_neu,residuen1_2)
#     plt.plot(datum_zeit_neu, Sigma_1_pos, color = 'r', label = "3-Sigma")
#     plt.plot(datum_zeit_neu, Sigma_1_neg, color = 'r')
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Detektion von Ausreißern Zeitreihe 1")
#     plt.legend () 
# =============================================================================

# =============================================================================
#     #Reihe 2 interpolierte Zeitreihe
#     plt.figure (figsize=(18,10))
#     #plt.plot (datum_zeit_neu, reihe2)
#     #plt.plot(datum_zeit, pol2)
#     plt.plot(datum_zeit,residuen2)
#     #plt.plot(datum_zeit_neu, reihe2_interpoliert)
#     plt.plot(datum_zeit_neu, Sigma_2_pos)
#     plt.plot(datum_zeit_neu, Sigma_2_neg)
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Zeitreihe 2")
#     plt.legend () 
# =============================================================================

# =============================================================================
#     #Detektion von Ausreißern Reihe 2
#     plt.figure (figsize=(18,10))
#     plt.plot(datum_zeit_neu,residuen2_2)
#     plt.plot(datum_zeit_neu, Sigma_2_pos, color = 'r', label = "3-Sigma")
#     plt.plot(datum_zeit_neu, Sigma_2_neg, color = 'r')
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Detektion von Ausreißern Zeitreihe 2")
#     plt.legend ()     
# =============================================================================

# =============================================================================
#     #Reihe 1 Filter
#     plt.figure (figsize=(18,10))
#     plt.plot(datum_zeit_neu, residuen1_2, label = "Zeitreihe 1")
#     plt.plot (datum_zeit_neu, filter1, label = "Gefilterte Zeitreihe 1")  
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Zeitreihe 1 gefiltert")
#     plt.legend ()     
#     
#     #Reihe 2 Filter
#     plt.figure (figsize=(18,10))
#     plt.plot(datum_zeit_neu, residuen2_2, label = "Zeitreihe 2")
#     plt.plot (datum_zeit_neu, filter2, label = "Gefilterte Zeitreihe 2")  
#     plt.grid ()
#     plt.xlabel ("Zeit")
#     plt.ylabel ("Verschiebung [mm]")
#     plt.title ("Zeitreihe 2 gefiltert")
#     plt.legend () 
# =============================================================================

# =============================================================================
#     #Reihe 1 und 2 FFT
#     plt.figure (figsize=(18,10))
#     plt.plot(frequenz1, amplitude1, color = 'r', label = "Zeitreihe 1")
#     plt.plot(frequenz_peak1_1,amplitude_peak_1_1,'o', color = 'r')
#     plt.plot(frequenz_peak1_2,amplitude_peak_1_2,'o', color = 'r')
#     plt.plot (frequenz2, amplitude2, color = 'g', label = "Zeitreihe 2")  
#     plt.plot(frequenz_peak2_1,amplitude_peak_2_1,'o', color = 'g')
#     plt.plot(frequenz_peak2_2,amplitude_peak_2_2,'o', color = 'g')
#     plt.xlim(0, 1/10000) 
#     plt.grid ()
#     plt.xlabel ("Frequenz [Hz]")
#     plt.ylabel ("Amplitude")
#     plt.title ("Amplitudendiagramm der Zeitreihen 1 und 2")
#     plt.legend ()     
# =============================================================================
    

# =============================================================================
#     #Reihe 1 Autokorrelationsfunktion
#     plt.figure (figsize=(18,10))
#     plt.plot (autokorr1_2[0], autokorr1_2[1])   
#     plt.xlim(0, None)
#     plt.grid ()
#     plt.xlabel ("Verschiebung [mm]")
#     plt.ylabel ("Korrelation")
#     plt.title ("Zeitreihe 1 Autokorrelationsfunktion")
#     #plt.legend () 
#     
#     #Reihe 2 Autokorrelationsfunktion
#     plt.figure (figsize=(18,10))
#     plt.plot (autokorr2_2[0], autokorr2_2[1]) 
#     plt.xlim(0, None)
#     plt.grid ()
#     plt.xlabel ("Verschiebung [mm]")
#     plt.ylabel ("Korrelation")
#     plt.title ("Zeitreihe 2 Autokorrelationsfunktion")
#     #plt.legend () 
# =============================================================================
    
# =============================================================================
#     #Reihe 1 und 2 Kreuzkorrelationsfunktion ungefiltert
#     plt.figure (figsize=(18,10))
#     plt.plot (kreuzkorr12[0], kreuzkorr12[1])  
#     plt.axvline(x=max_korr_x,color='r')
#     plt.plot(max_korr_x, max_korr_y,'o', color = 'r', label = "Maximaler Korrelationskoeffizient")
#     plt.grid ()
#     plt.xlabel ("Verschiebung [mm]")
#     plt.ylabel ("Korrelation")
#     plt.title ("Zeitreihe 1 und 2 Kreuzkorrelationsfunktion")
#     plt.legend () 
# =============================================================================

# =============================================================================
#     #Reihe 1 und 2 Kreuzkorrelationsfunktion gefiltert
#     plt.figure (figsize=(18,10))
#     plt.plot (kreuzkorr12_2[0], kreuzkorr12_2[1])  
#     plt.axvline(x=max_korr_x_2,color='r')
#     plt.plot(max_korr_x_2, max_korr_y_2,'o', color = 'r', label = "Maximaler Korrelationskoeffizient")
#     plt.grid ()
#     plt.xlabel ("Verschiebung [mm]")
#     plt.ylabel ("Korrelation")
#     plt.title ("Zeitreihe 1 und 2 Kreuzkorrelationsfunktion")
#     plt.legend () 
# =============================================================================
    
# =============================================================================
#     #Reihe 1 und 2 Kreuzkorrelationsfunktion Vergleich ungefiltert und gefiltert
#     plt.figure (figsize=(18,10))
#     plt.plot (kreuzkorr12[0], kreuzkorr12[1], color = 'r', label = "Kreuzkorrelation ungefiltert")  
#     plt.plot(max_korr_x, max_korr_y,'o', color = 'r')
#     plt.plot (kreuzkorr12_2[0], kreuzkorr12_2[1], color = 'g', label = "Kreuzkorrelation gefiltert")  
#     plt.plot(max_korr_x_2, max_korr_y_2,'o', color = 'g')
#     plt.xlim(-2000, 2000)
#     plt.grid ()
#     plt.xlabel ("Verschiebung [mm]")
#     plt.ylabel ("Korrelation")
#     plt.title ("Maximaler Korrelationskoeffizient")
#     plt.legend () 
# =============================================================================

