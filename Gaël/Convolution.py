import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt



def conv_fft(signal, impulse_response):
    """
    Code utilisant la FFT pour calculer la convolution.

    :param signal: Signal audio d'entrée (array NumPy)
    :param impulse_response: Réponse impulsionnelle (array NumPy)
    :return: Signal audio convolé
    """

    # Conserve le type de données du signal d'entrée
    dtype = str(signal.dtype)

    # Calculer la longueur de la convolution (signal + IR - 1)
    # C'est la taille nécessaire pour le zero-padding
    N = len(signal) + len(impulse_response) - 1

    # Calculer la FFT du signal et de la réponse impulsionnelle avec un zéro-padding
    # L'argument n=N assure le padding jusqu'à la longueur N
    X = np.fft.rfft(signal, n=N)
    H = np.fft.rfft(impulse_response, n=N)

    # Convolution dans le domaine fréquentiel (multiplication point par point)
    # C'est l'étape clé : F(f * h) = F(f) * F(h)
    Y = X * H

    # Appliquer la FFT inverse pour revenir au domaine temporel
    y = np.fft.irfft(Y)

    # Retourner la partie du signal qui correspond à la longueur du signal original
    # et s'assurer que le type de données est conservé.
    return y[:len(signal)].astype(dtype)

#TEST

fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lenght_sec = 47998

extrait = x[:34*lenght_sec]
extrait_conv = conv_fft(extrait, extrait)
sd.play(extrait_conv, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()