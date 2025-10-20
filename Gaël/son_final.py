import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

from Convolution import conv_fft
from RingModulation import ring_modulation
from Tremolo import tremolo
import instruments as i
import Convolution as c
from Seuillage import seuillage
from Egalisateur6bandes import egalisateur


fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lenght_sec = 47998

extrait = x[5*lenght_sec:34*lenght_sec]

signal = np.block([tremolo(extrait[:7*lenght_sec], 44100, 0.04, 1), ring_modulation(egalisateur(extrait[7*lenght_sec: 34*lenght_sec],0,0,0,1,1,1), 44100, 200)])
sd.play(signal, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()



