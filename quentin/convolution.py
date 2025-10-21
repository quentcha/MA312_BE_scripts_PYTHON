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
fe, data= wavfile.read(r'C:\Users\quent\OneDrive\Desktop\IPSA\python A3\BE_MA312_Python\MA312_BE_scripts_PYTHON\Gaël\Série_de_Fourier_BE_Ma312_2025.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])

convoluted_data=convolution(data,data)

print("plotting...")
plt.plot(np.abs(np.fft.rfft(convoluted_data)))
plt.plot(np.abs(np.fft.rfft(data)))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

print("playing")
play.sound(convoluted_data,fe)
