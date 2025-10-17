from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import time
'''
################################## Q.1
fe, data= wavfile.read('20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
print(data)


i=np.linspace(0, len(data), len(data))
print(len(data), fe)
plt.plot(i, data)
plt.show()

sd.play(data, fe)
time.sleep(len(data) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()
'''
################################## Q.2
def f(amplitude,t,freq):
    return amplitude*np.sin(2*np.pi*t*freq)
duree=2
fe =44100
freq =440
amplitude =0.8
n=6

t=np.linspace(0 , duree , fe*duree) # ou la version avec arange : np.arange(0, duree ,1/fe ,dtype = np.float32)
#signal=np.zeros(len(t))
#for i in range(1,n):
signal=f(amplitude,t,freq)+f((1/2)*amplitude,t,2*freq)+f((1/3)*amplitude,t,3*freq)+f((1/4)*amplitude,t,4*freq)+f((1/5)*amplitude,t,5*freq)
plt.plot(t, signal)
plt.show()

sd.play(signal, fe)
time.sleep(len(signal) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()
'''
######################Q.5
fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
#data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
#plt.plot(data)
#plt.show()
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
print(np.log2(len(data)))

spectre = np.fft.rfft(data)
freq=np.fft.rfftfreq(data.size, d=1./fe )
spectre_abs=np.abs(spectre)
plt.plot(freq, spectre_abs)
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

###############Q.4.5
fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
#data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation

data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
fmin,fmax=300,1000
spectre = np.fft.rfft(data)[fmin:fmax]
plt.plot(np.abs(spectre))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()
'''
