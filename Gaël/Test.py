import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from RingModulation import ring_modulation
from Egalisateur6bandes import egalisateur



fe, x = wavfile.read('Série_de_Fourier_BE_Ma312_2025.wav')
x = x.astype(np.float32)
if x.ndim == 2:
      x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)



lenght_sec = 47998
extrait = x[7*lenght_sec:34*lenght_sec]

extrait = ring_modulation(egalisateur(extrait[7*lenght_sec:],0,0,0,1,1,1), 44100, 200)
spectre = np.fft.rfft(extrait)                    # On calcule le spectre du signal avec la transformée de Fourier
freq = np.fft.rfftfreq(len(extrait), d=1.0/fe)

plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.plot(freq, np.abs(spectre), "r")
plt.show()

sd.play(extrait, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()