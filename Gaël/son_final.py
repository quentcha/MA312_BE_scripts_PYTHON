import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from shift import pitch
from RingModulation import ring_modulation
from Tremolo import tremolo
import instruments as i
from Seuillage import seuillage
from Egalisateur6bandes import egalisateur


fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lenght_sec = 47998

extrait = x[5*lenght_sec:34*lenght_sec]
extrait1 = tremolo(extrait[:7*lenght_sec], 44100, 0.035, 1)
extrait2 = pitch(ring_modulation(egalisateur(extrait[7*lenght_sec: int(12.4*lenght_sec)],1,1,2,4,4,4), 44100, 100), 100, 44100)
extrait3 = pitch(ring_modulation(egalisateur(extrait[int(12.4 *lenght_sec): int(17.95* lenght_sec) ],5,5,3,0,0,0), 44100, 100), 100, 44100)
extrait4 = pitch(ring_modulation(egalisateur(extrait[int(17.95*lenght_sec): 34*lenght_sec],2,2,2,4,4,4), 44100, 200), 200, 44100)
signal = np.block([ extrait1, extrait2, extrait3, extrait4])
sd.play(signal, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()



