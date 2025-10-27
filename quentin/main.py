from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import time

import play

from coupeBande import passe_coupe_bande

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])#0-padding pour une transfo de Fourier simple efficace

data_coupe=passe_coupe_bande(0,5000,data)
play.sound(data_coupe,fe)

plt.plot(np.abs(data_coupe))
plt.show()








#%% TEST Egalisateur à 6 bandes
fe, x = wavfile.read("20-20_000-Hz-Audio-Sweep.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

egalisateur(x, 1,1,1,1,2,2)