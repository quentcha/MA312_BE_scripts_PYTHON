import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import time
import matplotlib.pyplot as plt

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data /= (np.max(np.abs(data)) + 1*10**(-12))
sd.play(data, fe)
time.sleep(len(data) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()
