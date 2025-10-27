import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def coupe_bande(f,data,freq):
    spectre = np.fft.rfft(data)
    spectre_coupe= np.copy(spectre)
    n=1
    for i in range(len(freq)):
        if freq[i]==f*n:
            n+=1
            spectre_coupe[i]=0
    return np.fft.irfft(spectre_coupe)

fe, data= wavfile.read('20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])

freq = np.fft.rfftfreq(len(f(t,100)), d=1.0/fe)
