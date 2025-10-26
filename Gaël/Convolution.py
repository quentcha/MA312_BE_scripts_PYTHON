import numpy as np
import play
from scipy.io import wavfile
import matplotlib.pyplot as plt

def convolution(dataA, dataB):
    dataA, dataB=np.fft.rfft(dataA), np.fft.rfft(dataB)
    if len(dataB)>len(dataA):dataA=np.block([dataA,np.zeros(len(dataB)-len(dataA))])
    elif len(dataA)>len(dataB):dataB=np.block([dataB,np.zeros(len(dataA)-len(dataB))])
    dataC=dataA*dataB
    return np.fft.irfft(dataC)


def f(t,freq):
    return 5*np.sin(np.pi*t*freq)
def g(t,freq1,freq2):
    return 5*np.sin(np.pi*t*freq1)+5*np.sin(np.pi*t*freq2)

fe, data= wavfile.read('Série_de_Fourier_BE_Ma312_2025.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])

fe=30
duree=2
t=np.linspace(0 , duree , fe*duree)

convoluted_data=convolution(f(t,1000),g(t,1000,2300))

print("plotting...")
freq = np.fft.rfftfreq(len(f(t,100)), d=1.0/fe)
plt.stem(freq, np.abs(np.fft.rfft(convoluted_data)), linefmt='r-', markerfmt='ro', basefmt='r-')
plt.stem(freq, np.abs(np.fft.rfft(f(t,1000))), linefmt='g-', markerfmt='go', basefmt='g-')
plt.stem(freq, np.abs(np.fft.rfft(f(t,1200))), linefmt='b-', markerfmt='bo', basefmt='b-')
plt.xlabel("Frequence , Hz ")
plt.ylabel("Amplitude")
plt.show()

print("plotting")
plt.plot(convoluted_data)
plt.show()
print("playing")
play.sound(f(t,1000),fe)
play.sound(convoluted_data,fe)
