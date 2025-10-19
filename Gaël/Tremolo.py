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
    son = np.zeros(len(signal), dtype = complex)
    t = np.linspace(0, len(signal)/sampling_rate, len(signal))

    son = depth * np.abs(np.sin(2*np.pi * t *ft) * signal)

    return son


#TEST
fe, x = wavfile.read("guitare1.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

x_tremolo = tremolo(x, 44100, 10, 1)

sd.play(x_tremolo, fe)
time.sleep(len(x_tremolo) / fe)  # permet d'écouter un son
sd.stop()