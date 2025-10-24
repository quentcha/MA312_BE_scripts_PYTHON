import numpy as np
import play
def pitch(data,shift,fe):##erreur va s'additionner car on arrondi et perte des hautes fréquences
    spectre = np.fft.rfft(data)
    freq=np.fft.rfftfreq(data.size, d=1./fe)
    pas=freq[1]-freq[0]
    if shift>0: spectre = np.block([np.zeros(int(abs(shift)/pas)),spectre])
    elif shift<0: spectre = spectre[int(abs(shift)/pas):]
    return np.fft.irfft(spectre,n=len(data))

from scipy.io import wavfile
import matplotlib.pyplot as plt

#fe, data= wavfile.read(r'C:\Users\quent\OneDrive\Desktop\IPSA\python A3\BE_MA312_Python\MA312_BE_scripts_PYTHON\Série_de_Fourier_BE_Ma312_2025.wav')
#data = data.astype(np.float32)
#if data.ndim == 2 : #stéréo -> mono si besoin
#    data = data.mean(axis =1)

fe, data= wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)

data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
#play.sound(data,fe)
data_pitched=pitch(data,500,fe)
freq_pitched=np.fft.rfftfreq(data.size, d=1./fe)
#play.sound(data_pitched,fe)
print("plotting...")
plt.plot(freq_pitched,np.abs(np.fft.rfft(data)))
plt.plot(freq_pitched,np.abs(np.fft.rfft(data_pitched)))

plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

plt.plot(data_pitched)
plt.show()

#play.sound(data,fe)
#play.sound(data_pitched,fe)
#wavfile.write("example.wav", fe, data_pitched)
