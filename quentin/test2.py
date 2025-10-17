from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import play

def pitch_shift_fft(signal, shift_factor):
    N = len(signal)
    spec = np.fft.rfft(signal)
    new_spec = np.zeros_like(spec)

    # Décalage fréquentiel
    k_max = int(len(spec) / shift_factor)
    for k in range(k_max):
        new_k = int(k * shift_factor)
        if new_k < len(spec):
            new_spec[new_k] = spec[k]

    # Retour au temps
    shifted = np.fft.irfft(new_spec, n=N)
    return shifted

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
play.sound(pitch_shift_fft(data,4),fe)

'''
##############################ECHO
import numpy as np
import play
def pitch(data,n_octave):#pitch shift pur
    #freq*2 pour +1 octave
    spectre = np.fft.rfft(data)
    #pad=np.zeros(len(spectre)*2*n_octave,dtype=complex)
    #spectre=np.block([spectre[:(len(spectre)-len(pad))]])
    spectre_pitch=np.zeros(len(spectre)*(2**n_octave), dtype=complex)
    for i in range(len(spectre)):
        if i//(2**n_octave) < len(spectre):
            spectre_pitch[i]=spectre[i//(2**n_octave)]
    inverse=np.fft.irfft(spectre_pitch)
    return inverse

from scipy.io import wavfile
import matplotlib.pyplot as plt

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
#play.sound(data,fe)
data_pitched=pitch(data,3)
play.sound(data_pitched,fe)
print(data_pitched)
print("plotting...")
print(np.abs(np.fft.rfft(data_pitched)))
plt.plot(np.abs(np.fft.rfft(data_pitched)))
#plt.plot(np.abs(np.fft.rfft(data)))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

plt.plot(data_pitched)
plt.show()
'''
