import numpy as np
import play
def freq():
    pass
def pitch(data,n_octave):
    #freq*2 pour +1 octave
    spectre = np.fft.rfft(data)
    spectre_pitch=np.zeros(len(spectre)*(2*n_octave), dtype=complex)
    for i in range(len(spectre)):#source du problème ???
        spectre_pitch[i*2*n_octave]=spectre[i]
    return np.fft.irfft(spectre_pitch)

from scipy.io import wavfile
import matplotlib.pyplot as plt

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
#play.sound(data,fe)
#play.sound(pitch(data,3),fe)
data_pitched=pitch(data,3)

plt.plot(np.abs(np.fft.rfft(data_pitched)))
plt.plot(np.abs(np.fft.rfft(data)))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

plt.plot(data_pitched)
plt.show()
