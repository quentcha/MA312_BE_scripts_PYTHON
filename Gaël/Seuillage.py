import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt




def seuillage(data, fe,  thau, k): # Soit thau le seuil et k le coefficient de réduction




    spectre = np.fft.rfft(data)                    # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)

    spectre_filtre = []  # On crée une liste vide qui va accueillir le spectre filtré
    for i in range(len(spectre)):
        a = np.abs(spectre[i])
        if a < thau:
            spectre_filtre.append(k * a)
        else:
            spectre_filtre.append(a)

    son = np.fft.irfft(spectre_filtre)


    # On renvoie le plot du spectre initial et du spectre filtré pour les comparer:

    '''plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(freq, np.abs(spectre), "r")
    plt.plot(freq, np.abs(spectre_filtre), "g")
    plt.show()

    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:

    
    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    #sd.play(x,fe)
    #time.sleep(len(son) / fe)  # permet d'écouter un son
    #sd.stop()'''
    return son
 # TEST
'''fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lenght_sec = 47998

extrait = x[5*lenght_sec:34*lenght_sec]
extrait = seuillage(extrait, 44100, 1000, 0)

sd.play(extrait, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()'''


