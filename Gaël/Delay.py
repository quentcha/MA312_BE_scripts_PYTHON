import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def delay(signal, sampling_rate, delay_time=0.5, feedback=0.5, mix=0.5):
    """
    Applique un effet de délai sur le signal.

    :param signal: Le signal audio d'entrée
    :param sampling_rate: Le taux d'échantillonnage (Hz)
    :param delay_time: Temps de retard en secondes entre les échos
    :param feedback: Rétroaction, détermine l'amplitude des échos (si en dessous de 1, alors l'amplitude de l'écho décroit à chaque répétition)
    :param mix: Mixage du signal traité (entre 0 et 1), caractérise l'équilibre entre le signal et les échos ( 0 = signal original, 1 = uniquement les échos)
    :return: Signal avec l'effet de délai appliqué
    """

    dtype = str(signal.dtype)

    # Convertit le temps de délai en nombre d'échantillons
    delay_samples = int(delay_time * sampling_rate)

    # Crée le tableau de sortie plus grand pour accueillir le délai (echo)
    output = np.zeros(len(signal) + delay_samples, dtype=np.float64)

    # Boucle d'application de l'effet de délai (avec feedback)
    for i in range(len(signal)):
        # Ajout du signal original à l'index actuel
        output[i] += signal[i]

        # Si nous sommes au-delà du temps de délai, ajoute l'écho
        if i >= delay_samples:
            # Rétroaction = l'écho de la valeur précédente * facteur de feedback
            output[i] += feedback * output[i - delay_samples]

    # Mixage : Combine le signal original avec la partie du signal de sortie qui chevauche
    # (le signal traité est tronqué à la longueur du signal original pour le mix)
    # C'est l'étape qui permet de choisir combien de signal original (1-mix) et combien de signal traité (mix) sont conservés.
    output = (1 - mix) * signal + mix * output[:len(signal)]

    # Retourne le signal final avec le type de données original
    return output.astype(dtype)

#TEST
fe, x = wavfile.read("sample1.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

x_delay = delay(x, 44100, 0.7, 0.5, 0.5)

sd.play(x_delay, fe)
time.sleep(len(x_delay) / fe)  # permet d'écouter un son
sd.stop()