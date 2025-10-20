import numpy as np
import play

def convolution(dataA, dataB):
    dataA, dataB=np.fft.rfft(dataA), np.fft.rfft(dataB)
    if len(dataB)>len(dataA):dataA=np.block([dataA,np.zeros(len(dataB)-len(dataA))])
    elif len(dataA)>len(dataB):dataB=np.block([dataB,np.zeros(len(dataA)-len(dataB))])
    dataC=dataA*dataB
    return np.fft.irfft(dataC)

from scipy.io import wavfile
import matplotlib.pyplot as plt
from instruments import note, hihat
def f(t,freq):
    return abs(5*t*np.sin(np.pi*t*freq))
def y(t,freq):
    return -abs(5*t*np.sin(np.pi*t*freq))
def g(t,freq):
    return 5*np.sin(2*np.pi*t*freq)
fe, data= wavfile.read('Série_de_Fourier_BE_Ma312_2025.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])

duree=5
t=np.linspace(0 , duree , fe*duree)

convoluted_data=convolution(data,data)

print("plotting...")
print(np.abs(np.fft.rfft(convoluted_data)))
plt.plot(np.abs(np.fft.rfft(convoluted_data)))
plt.plot(np.abs(np.fft.rfft(data)))
plt.plot(np.abs(np.fft.rfft(g(t,1000))))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

#plt.plot(convoluted_data)
#plt.show()
#plt.plot(f(t,261))
#plt.show()
#plt.plot(y(t,261))
#plt.show()

#play.sound(f(t,261),fe)
#play.sound(y(t,261),fe)

print("plotting")
plt.plot(convoluted_data)
plt.show()
print("playing")
#play.sound(g(t,1000),fe)
play.sound(convoluted_data,fe)
