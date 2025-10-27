import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import Librairie as lb

#%% TEST Egalisateur à 6 bandes

fe, x = wavfile.read("20-20_000-Hz-Audio-Sweep.wav")
x = x.astype(np.float32)                                #On importe notre son test
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lb.egalisateur(x, 1,1,1,1,2,2)  #On l'égalise pour faire ressortir les fréquences aigues (> 800 Hz)

#%% TEST Filtrage ampliude
lb.filtrage_amp("guitare1.wav", 2000, 8000)     #On garde sur le son de guitare les amplitudes entre 2000 et 8000
lb.filtrage_amp("Série_de_Fourier_BE_Ma312_2025.wav", 10000, 15000) #On garde sur le son des séries de Fourier uniquement les amplitudes entre 10000 et 15000

#%% TEST Seuillage

fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lb.seuillage(x, 44100, 10000, 0)

#%% TEST Tremolo
duree = 2
sampling_rate = 44100
nb_points = int(duree * sampling_rate)
t = np.linspace(0, duree, nb_points, endpoint=False)    #On définit notre vecteur temps

frequence_signal = 392
x = np.sin(2 * np.pi * t * frequence_signal)  #On crée une note sol

x_tremolo = lb.tremolo(x, sampling_rate, 0.5, 1)  # On applique le trémolo à notre note

plt.figure(figsize=(12, 5))
plt.title("Signal Original vs. Signal Trémolo (Vue générale)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")         #On affiche la forme d'onde de nos deux signaux pour les comparer

# On ne trace pas la valeur absolue pour les signaux**
plt.plot(t, x, "r", alpha=0.3, label="Signal Original (392 Hz)")
plt.plot(t, x_tremolo, "g", label="Signal Trémolo (Modulation à 0.5 Hz)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()

sd.play(x_tremolo, fe)
time.sleep(len(x_tremolo) / fe)  # On joue notre son
sd.stop()

#%% TEST Ring Modulation

fe, x = wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)        # On importe le son
lenght_sec = 47998
extrait = x[5*lenght_sec:34*lenght_sec]

x_rm = lb.ring_modulation(extrait, 44100, 50)
signal = np.block([x_rm])
sd.play(signal, fe)
time.sleep(len(signal) / fe)  # On écoute le son
sd.stop()


#On fait pareil avec une note continue
duree = 2
sampling_rate = 44100
nb_points = int(duree * sampling_rate)

t = np.linspace(0, duree, nb_points, endpoint=False)


frequence_signal = 392
x = np.sin(2 * np.pi * t * frequence_signal)


x_rm = lb.ring_modulation(x, sampling_rate, 200)


plt.figure(figsize=(12, 5))
plt.title("Signal Original vs. Signal Ring Modulation ")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")


# On ne trace pas la valeur absolue pour les signaux**
plt.plot(t, x, "r", alpha=0.3, label="Signal Original (392 Hz)")
plt.plot(t, x_rm, "g", label="Signal Ring_modulation (Modulation à 400 Hz)")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()

sd.play(x_rm, sampling_rate)
time.sleep(len(x_rm) / sampling_rate)  # permet d'écouter un son
sd.stop()

#%% TEST Passe Bande
fe, data= wavfile.read('20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
freq = np.fft.rfftfreq(len(data), d=1.0/fe)

fmin,fmax=500,15000
plt.plot(freq,np.abs(np.fft.rfft(data)))
plt.plot(freq,np.abs(np.fft.rfft(lb.passe_bande(fmin,fmax,data,freq))))
plt.xlabel("Frequence , Hz ")
plt.ylabel("Amplitude")
plt.show()

#%% TEST Frequency shift
fe, data= wavfile.read("Série_de_Fourier_BE_Ma312_2025.wav")
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
#play.sound(data,fe)
data_pitched=lb.pitch(data,500,fe)
freq_pitched=np.fft.rfftfreq(data_pitched.size, d=1./fe)
#play.sound(data_pitched,fe)
print("plotting...")
plt.plot(freq_pitched,np.abs(np.fft.rfft(data)))
plt.plot(freq_pitched,np.abs(np.fft.rfft(data_pitched)))

plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

plt.plot(data_pitched)
plt.show()
