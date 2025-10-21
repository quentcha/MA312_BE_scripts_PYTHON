import numpy as np
import sounddevice as sd
import time

def sound(data,fe):
    data_norm = data / (np.max(np.abs(data)) + 1e-12)
    sd.play(data_norm, fe)
    time.sleep(len(data_norm) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
    sd.stop()
'''

from scipy.io import wavfile
fe, data= wavfile.read(r'TechnoSerieDeFourier.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
sound(data,fe)

'''
