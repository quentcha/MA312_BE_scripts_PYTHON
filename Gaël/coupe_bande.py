import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def coupe_bande(fmin,fmax,data,freq):
    spectre = np.fft.rfft(data)
    spectre_coupe= np.copy(spectre)
    for i in range(len(freq)):
        if freq[i]>=fmin and freq[i]<=fmax:
            spectre_coupe[i]=0
    return np.fft.irfft(spectre_coupe)

fe, data= wavfile.read('20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
freq = np.fft.rfftfreq(len(data), d=1.0/fe)

fmin,fmax=500,500
plt.plot(freq,np.abs(np.fft.rfft(data)))
plt.plot(freq,np.abs(np.fft.rfft(coupe_bande(fmin,fmax,data,freq))))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()
