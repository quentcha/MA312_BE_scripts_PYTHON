import numpy as np
def passe_coupe_bande(fmin,fmax,data):
    #spectre = np.fft.rfft(data)[fmin:fmax]
    spectre = np.fft.rfft(data)
    spectre_coupe= np.zeros(len(spectre), dtype=complex)
    for i in range(len(spectre)):
        if i>fmin and i<fmax:
            spectre_coupe[i]=spectre[i]
    return np.fft.irfft(spectre_coupe)

'''
from scipy.io import wavfile
import matplotlib.pyplot as plt

fe, data= wavfile.read('20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
#data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation

data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
fmin,fmax=0,15000
plt.plot(np.abs(np.fft.rfft(data)))
plt.plot(np.abs(np.fft.rfft(passe_coupe_bande(fmin,fmax,data))))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()
'''



