import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def tremolo(signal, sampling_rate= 44100, ft=0.5, depth=1):
    """
    Applique un effet de trémolo sur le signal.
    Le trémolo est un effet sonore qui applique une fonction sinusoïdale
    strictement positive (car une amplitude ne peut pas être négative) à l'amplitude du signal d'entrée.
    Cela aura pour effet de diminuer puis rétablir l'instensité du son de façon périodique.

    :param signal: Le signal audio d'entrée
    :param sampling_rate: La fréquence d'échantillonnage (Hz)
    :param ft (fréquence du trémolo): Détermine la périodicité de l'effet, généralement on prend des fréquences très basses afin que l'effet soit prononcé ( entre 0.5 et 10 Hz)
    :param depth: détermine l'amplitude de l'effet
    """

    t = np.linspace(0, len(signal)/sampling_rate, len(signal))
    son = signal * (1 - depth * np.abs(np.cos(np.pi * t *ft)))
    return son


#TEST
'''
duree = 2
sampling_rate = 44100
nb_points = int(duree * sampling_rate)

t = np.linspace(0, duree, nb_points, endpoint=False)


frequence_signal = 392
x = np.sin(2 * np.pi * t * frequence_signal)


x_tremolo = tremolo(x, sampling_rate, 0.5, 1)


plt.figure(figsize=(12, 5))
plt.title("Signal Original vs. Signal Trémolo (Vue générale)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")


# On ne trace pas la valeur absolue pour les signaux**
plt.plot(t, x, "r", alpha=0.3, label="Signal Original (392 Hz)")
plt.plot(t, x_tremolo, "g", label="Signal Trémolo (Modulation à 0.5 Hz)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()



sd.play(x_tremolo, fe)
time.sleep(len(x_tremolo) / fe)  # permet d'écouter un son
sd.stop()'''