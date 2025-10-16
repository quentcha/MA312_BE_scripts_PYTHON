import numpy as np
from scipy.io import wavfile
import sounddevice as sd

"""On définit les 6 plages de fréquences auxquelles on appliquera les coefficients pris en compte dans la fonction:
Basse : 0-100 Hz

Basse-médium : 101-200 Hz

Médium-grave : 201-400 Hz

Médium-aigu : 401-800 Hz

Aigu : 0,801-1,6 kHz

Très aigu: 1,6-3,2 kHz"""

Très aigu : 3,2 kHz
def egalisateur(data, k1, k2, k3, k4, k5, k6): # Soient ki, les coefficients multiplicateurs des bandes définies

    fe, x = wavfile.read(data)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x /= (np.max(np.abs(x)) + 1e-12)


    spectre = np.fft.rfft(x)                    # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(x), d=1.0/fe)

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
    ta_index_max = np.searchsorted(freq, 3200)


    # On applique les facteurs à leur plage de fréquences:

    spectre_filtre = spectre.copy() # Créez une copie pour ne pas modifier l'original

    spectre_filtre[b_index_min:b_index_max] = k1 * spectre_filtre[b_index_min:b_index_max]
    spectre_filtre[bm_index_min:bm_index_max] = k2 * spectre_filtre[bm_index_min:bm_index_max]
    spectre_filtre[mg_index_min:mg_index_max] = k3 * spectre_filtre[mg_index_min:mg_index_max]
    spectre_filtre[bm_index_min:bm_index_max] = k2 * spectre_filtre[bm_index_min:bm_index_max]



    # --- 4. Reconstruire le signal avec la transformée inverse ---
    son_filtre = np.fft.irfft(spectre_filtre)

    # --- 5. Écouter le son reconstruit ---
    # Vous pouvez aussi l'enregistrer dans un fichier wav si vous préférez
    # wavfile.write('guitare_filtree.wav', fe, son_filtre.astype(np.int16))
    sd.play(son_filtre, fe)
    status = sd.wait() # Attendre que le son soit joué