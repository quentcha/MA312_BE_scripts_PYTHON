import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path

# Lecture du fichier
fichier = Path(r"Série_de_Fourier_BE_Ma312_2025.wav")
fe, data = wavfile.read(fichier)
data = data.astype(np.float32)
if data.ndim == 2:
    data = data.mean(axis=1)

def pitch(data,n_octave):#pitch shift pur
    #freq*2 pour +1 octave
    spectre = np.fft.rfft(data)
    spectre_pitch=np.zeros(len(spectre), dtype=complex)
    i_max=int(len(spectre)/(2**n_octave))
    for i in range(len(spectre)):
        index=int(i/(2**n_octave))
        if index < len(spectre):
            spectre_pitch[i]=spectre[index]

    inverse=np.fft.irfft(spectre_pitch,n=len(data))
    return inverse

#Application du pitch
data_pitched = pitch(data, 1)

# Calcul du spectre et de l’axe fréquentiel
freqs = np.fft.rfftfreq(len(data), d=1/fe)
fft_data = np.abs(np.fft.rfft(data))
fft_pitched = np.abs(np.fft.rfft(data_pitched))

#Tracé du spectre
plt.figure(figsize=(10,5))
plt.plot(freqs, fft_data, "g",  label="Original")
plt.plot(freqs, fft_pitched, "r", label="Décalé")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.legend()
#plt.grid(True)
plt.show()
