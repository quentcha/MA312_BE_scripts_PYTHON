import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt




def filtrage_amp(data, A1, A2): # Soient A1 et A2 les amplitudes des bornes

    fe, x = wavfile.read(data)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x /= (np.max(np.abs(x)) + 1e-12)


    spectre = np.fft.rfft(x)                    # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(x), d=1.0/fe)

    spectre_filtre = np.zeros(len(spectre), dtype = complex)     # On crée une liste vide qui va accueillir le spectre filtré
    for i in range(len(spectre)):
        a = np.abs(spectre[i])
        if a < A1 or a > A2:
            spectre_filtre.append(0)
        else:
            spectre_filtre.append(a)



    # On renvoie le plot du spectre initial et du spectre filtré pour les comparer:

    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(freq, np.abs(spectre), "r")
    plt.plot(freq, np.abs(spectre_filtre), "g")
    plt.grid(True)
    plt.show()
    '''
    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:

    son = np.fft.irfft(spectre_filtre)
    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    sd.play(x,fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()
    '''

 # TEST

#filtrage_amp("guitare1.wav", 2000, 8000)
filtrage_amp("Série_de_Fourier_BE_Ma312_2025.wav", 10000, 15000)