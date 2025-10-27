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

fe, data= wavfile.read('guitare1.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation

duree=5
t=np.linspace(0 , duree , fe*duree)
freq=np.fft.rfftfreq(data.size, d=1./fe)
convoluted_data=convolution(data,data)
convoluted_freq=np.fft.rfftfreq(convoluted_data.size, d=1./fe)
print("plotting...")
plt.plot(convoluted_freq,np.abs(np.fft.rfft(convoluted_data)))
plt.plot(freq,np.abs(np.fft.rfft(data)),'r')
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

print("plotting")
plt.title("Forme d'onde temporelle")
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.plot( convoluted_data, 'g', label="Son convolué")
plt.legend()
plt.show()

plt.title("Forme d'onde temporelle")
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.plot( data,'r', label = "Son original")
plt.legend()
plt.show()
print("playing")
play.sound(data,fe)
play.sound(convoluted_data,fe)
