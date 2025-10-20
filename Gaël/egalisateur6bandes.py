import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

"""On définit les 6 plages de fréquences auxquelles on appliquera les coefficients pris en compte dans la fonction:
Basse : 0-100 Hz

Basse-médium : 101-200 Hz

Médium-grave : 201-400 Hz

Médium-aigu : 401-800 Hz

Aigu : 0,801-1,6 kHz

Très aigu: 1,6-5 kHz"""


def egalisateur(data, k1, k2, k3, k4, k5, k6): # Soient ki, les coefficients multiplicateurs des bandes définies

    fe = 44100
    spectre = np.fft.rfft(data)                    # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)

    # On définit les indices des 6 plages de fréquence:
    b_index_min = np.searchsorted(freq, 0)         # Basse
    b_index_max = np.searchsorted(freq, 100)

    bm_index_min = np.searchsorted(freq, 101)      # Basse médium
    bm_index_max = np.searchsorted(freq, 200)

    mg_index_min = np.searchsorted(freq, 201)      #Médium grave
    mg_index_max = np.searchsorted(freq, 400)

    ma_index_min = np.searchsorted(freq, 401)      #Médium aigu
    ma_index_max = np.searchsorted(freq, 800)

    a_index_min = np.searchsorted(freq, 801)       #Aigu
    a_index_max = np.searchsorted(freq, 1600)

    ta_index_min = np.searchsorted(freq, 1601)     #Très aigu
    ta_index_max = np.searchsorted(freq, 5000)


    # On applique les facteurs à leur plage de fréquences:

    spectre_filtre = spectre.copy()

    spectre_filtre[b_index_min:b_index_max] = k1 * spectre_filtre[b_index_min:b_index_max]
    spectre_filtre[bm_index_min:bm_index_max] = k2 * spectre_filtre[bm_index_min:bm_index_max]
    spectre_filtre[mg_index_min:mg_index_max] = k3 * spectre_filtre[mg_index_min:mg_index_max]
    spectre_filtre[ma_index_min:ma_index_max] = k4 * spectre_filtre[ma_index_min:ma_index_max]
    spectre_filtre[a_index_min:a_index_max] = k5 * spectre_filtre[a_index_min:a_index_max]
    spectre_filtre[ta_index_min:ta_index_max] = k6 * spectre_filtre[ta_index_min:ta_index_max]

    # On renvoie le plot du spectre initial et du spectre égalisé pour pouvoir les comparer:
    #plt.xlabel("Fréquence (Hz)")
    #plt.ylabel("Amplitude")
    #plt.plot(freq, np.abs(spectre), "g")
    #plt.plot(freq, np.abs(spectre_filtre), "r")
    #plt.show()

    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:

    '''son = np.fft.irfft(spectre_filtre)
    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    sd.play(x,fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()'''
    son = np.fft.irfft(spectre_filtre)
    return son

 # TEST

#egalisateur("guitare1.wav", 10, 10, 1, 1, 1, 1)
#egalisateur("Série_de_Fourier_BE_Ma312_2025.wav", 10, 10, 0.5, 0.5, 0.5, 0.5)
'''fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lenght_sec = 47998

extrait = x[7*lenght_sec:34*lenght_sec]
extrait = egalisateur(extrait, 0, 0, 0, 1, 1, 1)

sd.play(extrait, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()'''