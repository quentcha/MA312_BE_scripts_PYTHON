from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import time

import play

from coupeBande import passe_coupe_bande

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])#0-padding pour une transfo de Fourier simple efficace

data_coupe=passe_coupe_bande(0,5000,data)
play.sound(data_coupe,fe)

plt.plot(np.abs(data_coupe))
plt.show()


def egalisateur(data, k1, k2, k3, k4, k5, k6):  # Soient ki, les coefficients multiplicateurs des bandes définies

    """
    Cette fonction permet d'amplifier ou de réduire l'amplitude de fréquences ciblées sur un signal sonore.
    On définit les 6 plages de fréquences auxquelles on appliquera les coefficients pris en compte dans la fonction:

    Basse : 0-100 Hz

    Basse-médium : 101-200 Hz

    Médium-grave : 201-400 Hz

    Médium-aigu : 401-800 Hz

    Aigu : 0,801-5 kHz

    Très aigu: 5-20 kHz"""

    fe = 44100
    spectre = np.fft.rfft(data)  # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(data), d=1.0 / fe)

    # On définit les indices des 6 plages de fréquence:
    b_index_min = np.searchsorted(freq, 0)  # Basse
    b_index_max = np.searchsorted(freq, 100)

    bm_index_min = np.searchsorted(freq, 101)  # Basse médium
    bm_index_max = np.searchsorted(freq, 200)

    mg_index_min = np.searchsorted(freq, 201)  # Médium grave
    mg_index_max = np.searchsorted(freq, 400)

    ma_index_min = np.searchsorted(freq, 401)  # Médium aigu
    ma_index_max = np.searchsorted(freq, 800)

    a_index_min = np.searchsorted(freq, 801)  # Aigu
    a_index_max = np.searchsorted(freq, 5000)

    ta_index_min = np.searchsorted(freq, 5000)  # Très aigu
    ta_index_max = np.searchsorted(freq, 20000)

    # On applique les facteurs à leur plage de fréquences:

    spectre_filtre = spectre.copy()

    spectre_filtre[b_index_min:b_index_max] = k1 * spectre_filtre[b_index_min:b_index_max]
    spectre_filtre[bm_index_min:bm_index_max] = k2 * spectre_filtre[bm_index_min:bm_index_max]
    spectre_filtre[mg_index_min:mg_index_max] = k3 * spectre_filtre[mg_index_min:mg_index_max]
    spectre_filtre[ma_index_min:ma_index_max] = k4 * spectre_filtre[ma_index_min:ma_index_max]
    spectre_filtre[a_index_min:a_index_max] = k5 * spectre_filtre[a_index_min:a_index_max]
    spectre_filtre[ta_index_min:ta_index_max] = k6 * spectre_filtre[ta_index_min:ta_index_max]

    # On renvoie le spectre initial et le spectre égalisé afin de les comparer:
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(freq, np.abs(spectre), "g")
    plt.plot(freq, np.abs(spectre_filtre), "r")
    plt.show()

    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:
    son = np.fft.irfft(spectre_filtre)
    '''son = np.fft.irfft(spectre_filtre)
    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    sd.play(x,fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()'''

    return son





#%% TEST Egalisateur à 6 bandes
fe, x = wavfile.read("20-20_000-Hz-Audio-Sweep.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

egalisateur(x, 1,1,1,1,2,2)