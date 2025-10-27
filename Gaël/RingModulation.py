import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt



def ring_modulation(signal, sampling_rate= 44100, fp=400.0):
    """
    Applique l'effet Ring Modulation.

    Le signal d'entrée est multiplié par une onde porteuse sinusoïdale
    à la fréquence 'carrier_freq'.

    :param signal: Le signal audio d'entrée
    :param sampling_rate: Le taux d'échantillonnage (Hz)
    :param fp: Fréquence de la porteuse (Hz, souvent audible)
    :return: Signal avec l'effet de modulation en anneau appliqué
    """


    t = np.linspace(0, len(signal) / sampling_rate, len(signal))

    # On crée l'onde porteuse (Carrier Wave)
    carrier_wave = np.sin(2 * np.pi * fp * t)

    # On appliquer la Modulation en Anneau
    signal_traite = signal * carrier_wave

    # Normalisation pour éviter la saturation après la multiplication
    max_val = np.max(np.abs(signal_traite))
    if max_val > 0:
        signal_traite /= max_val

    return signal_traite
'''
#TEST

fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)
lenght_sec = 47998
extrait = x[5*lenght_sec:34*lenght_sec]

x_rm = ring_modulation(extrait, 44100, 50)
signal = np.block([x_rm])
signal = np.block([signal, signal])
sd.play(signal, fe)
time.sleep(len(signal) / fe)  # permet d'écouter un son
sd.stop()
'''

'''
duree = 2
sampling_rate = 44100
nb_points = int(duree * sampling_rate)

t = np.linspace(0, duree, nb_points, endpoint=False)


frequence_signal = 392
x = np.sin(2 * np.pi * t * frequence_signal)


x_rm = ring_modulation(x, sampling_rate, 200)


plt.figure(figsize=(12, 5))
plt.title("Signal Original vs. Signal Ring Modulation ")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")


# On ne trace pas la valeur absolue pour les signaux**
plt.plot(t, x, "r", alpha=0.3, label="Signal Original (392 Hz)")
plt.plot(t, x_rm, "g", label="Signal Ring_modulation (Modulation à 400 Hz)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()

sd.play(x_rm, sampling_rate)
time.sleep(len(x_rm) / sampling_rate)  # permet d'écouter un son
sd.stop()
'''