import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import Librairie as lb

#%% TEST Egalisateur à 6 bandes

fe, x = wavfile.read("20-20_000-Hz-Audio-Sweep.wav")
x = x.astype(np.float32)                                #On importe notre son test
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lb.egalisateur(x, 1,1,1,1,2,2)  #On l'égalise pour faire ressort les fréquences aigues (> 800 Hz)

#%% TEST Filtrage ampliude
lb.filtrage_amp("guitare1.wav", 2000, 8000)     #On garde sur le son de guitare les amplitudes entre 2000 et 8000
lb.filtrage_amp("Série_de_Fourier_BE_Ma312_2025.wav", 10000, 15000) #On garde sur le son des séries de Fourier uniquement les amplitudes entre 10000 et 15000