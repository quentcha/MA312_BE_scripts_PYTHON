import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import time

def sound(data,fe):
    data_norm = data / (np.max(np.abs(data)) + 1e-12)
    sd.play(data_norm, fe)
    time.sleep(len(data_norm) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
    sd.stop()

def pitch_shift_no_interp(data, n_octave):
    """Décale la hauteur selon le changement d’échelle, sans interpolation."""
    factor = 2 ** n_octave  # 2 pour +1 octave, 0.5 pour -1
    N = len(data)

    # FFT directe
    spectre = np.fft.rfft(data)
    freq = np.fft.rfftfreq(N)

    # Échelle fréquentielle
    new_len = int(len(spectre) / factor)
    spectre_shifted = np.zeros_like(spectre, dtype=complex)
    spectre_shifted[:new_len] = spectre[:new_len] * factor  # amplification du module (1/|a|)

    # Retour temporel
    data_shifted = np.fft.irfft(spectre_shifted)

    # Comme on change d’échelle temporelle, la fréquence d’échantillonnage change aussi
    return data_shifted, factor

# Exemple d’usage :
fe, data = wavfile.read("guitare1.wav")
sound(data,fe)
data = data.mean(axis=1) if data.ndim == 2 else data
data = data.astype(np.float32)

y, facteur = pitch_shift_no_interp(data, n_octave=3)  # +1 octave
sound(y,int(fe*facteur))
