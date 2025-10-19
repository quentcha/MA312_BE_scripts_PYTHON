
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

#%% Q1
fe, x= wavfile.read('Série_de_Fourier_BE_Ma312_2025.wav')
x = x.astype(np.float32)
if x.ndim == 2 : #stéréo -> mono si besoin
    x = x.mean(axis =1)
x /= (np.max(np.abs(x))+1*10**(-12))#normalisation

sd.play(x,fe)
time.sleep(len(x)/fe)   # permet d'écouter un son
sd.stop()

#%% Q2
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
fe = 44000
duree = 3 #secondes
monenregistrement= sd.rec(int(duree * fe), samplerate=fe, channels =1)   # sert à enregistrer
sd.wait()
sd.play(monenregistrement,fe)
time.sleep(len(monenregistrement)/fe)   # permet d'écouter un son
sd.stop()

plt.plot(monenregistrement)
plt.show()




#%% Q3
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

amplitude = 0.05
freq = 5*440
duree = 2
fe = 5000
t = np.linspace(0, duree, fe * duree)
signal = 1 - np.abs(t)

plt.plot(t, signal)
plt.title("Représentation du signal sonore")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()  # Affiche la fenêtre du graphique

sd.play(signal, fe)
time.sleep(len(signal) / fe)
sd.stop()

#%% Q4
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

tempo = 1/4
amplitude = 1
echantillonage = 44100
temps = tempo / 4
pause = np.linspace(0, temps, int(echantillonage * temps))
freq = 262 # do
temps = tempo / 2
t = np.linspace(0, temps, int(echantillonage * temps))
note = amplitude * np.sin(2 * np.pi * t * freq)
signal = np.block([note, pause, note])
freq = 392 # sol
temps = tempo / 2
t = np.linspace(0, temps, int(echantillonage * temps))
note = amplitude * np.sin(2 * np.pi * t * freq)
signal = np.block([signal, pause, note, pause, note])

sd.play(signal, echantillonage)
time.sleep(len(signal) / echantillonage)
sd.stop()

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


tempo = 1/4
amplitude = 1
echantillonage = 44100
temps = tempo / 4
pause = np.linspace(0, temps, int(echantillonage * temps))
temps = tempo / 2
t = np.linspace(0, temps, int(echantillonage * temps))

sol = amplitude * np.sin(2 * np.pi * t * 392)
do =  amplitude * np.sin(2 * np.pi * t * 262)
rediez = amplitude * np.sin(2 * np.pi * t * 311.13)
soldiez = amplitude * np.sin(2 * np.pi * t * 415.30)
fa = amplitude * np.sin(2 * np.pi * t * 349.23)
re = amplitude * np.sin(2 * np.pi * t * 293.66)

signal = np.block([sol, pause, sol, pause, do, pause, rediez, pause, rediez, pause, pause, pause, soldiez, pause, soldiez, pause, pause, pause, re,pause, fa, pause, fa, pause])
signal = np.block([signal, signal, signal, signal, signal])

sd.play(signal, echantillonage)
time.sleep(len(signal) / echantillonage)
sd.stop()
sd.stop()

#%% Q5
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math

fe, x= wavfile.read('guitare1.wav')
x = x.astype(np.float32)
if x.ndim == 2 : #stéréo -> mono si besoin
    x = x.mean(axis =1)
x /= (np.max(np.abs(x))+1*10**(-12))#normalisation
t = np.linspace(0, len(x)/fe, len(x))

sd.play(x,fe)
time.sleep(len(x)/fe)   # permet d'écouter un son
sd.stop()

plt.plot(t, x)
plt.show()
taille = math.floor((len(x)**(0.5) - 1)**2)
print(taille)

#%% Q6
import numpy as np
from scipy.io import wavfile
import sounddevice as sd # Module pour jouer le son
import matplotlib.pyplot as plt

# --- 1. Charger et préparer le signal (comme dans vos scripts précédents) ---
fe, x = wavfile.read('guitare1.wav')
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

# --- 2. Calculer le spectre du signal ---
spectre = np.fft.rfft(x)
freq = np.fft.rfftfreq(len(x), d=1.0/fe)

# --- 3. Filtrer le spectre : supprimer des fréquences ---
# Définir les fréquences de coupure
frequence_min = 200  # Par exemple, on enlève tout ce qui est en dessous de 200 Hz
frequence_max = 1000 # On enlève tout ce qui est au-dessus de 1000 Hz

# Trouver les indices correspondants à ces fréquences
index_min = np.searchsorted(freq, frequence_min)
index_max = np.searchsorted(freq, frequence_max)

# Mettre à zéro les coefficients du spectre hors de la plage d'intérêt
spectre_filtre = spectre.copy() # Créez une copie pour ne pas modifier l'original
spectre_filtre[:index_min] = 0 # Met à zéro les basses fréquences
spectre_filtre[index_max:] = 0 # Met à zéro les hautes fréquences

# --- 4. Reconstruire le signal avec la transformée inverse ---

# --- 5. Écouter le son reconstruit ---
# Vous pouvez aussi l'enregistrer dans un fichier wav si vous préférez
# wavfile.write('guitare_filtree.wav', fe, son_filtre.astype(np.int16))
sd.play(son_filtre, fe)
status = sd.wait() # Attendre que le son soit joué

#%% Q6.2

def f(data, fe, fnote):
    fe, x = wavfile.read(data)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x /= (np.max(np.abs(x)) + 1e-12)

    # --- 2. Calculer le spectre du signal ---
    spectre = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), d=1.0 / fe)


    # Trouver les indices correspondants à ces fréquences
    index = np.searchsorted(freq, fnote)


    # Mettre à zéro les coefficients du spectre hors de la plage d'intérêt
    spectre_filtre = spectre.copy()  # Créez une copie pour ne pas modifier l'original
    spectre_filtre[:index-1] = 0  # Met à zéro les basses fréquences
    spectre_filtre[index+1:] = 0  # Met à zéro les hautes fréquences

    # --- 4. Reconstruire le signal avec la transformée inverse ---
    son_filtre = np.fft.irfft(spectre_filtre)

    # --- 5. Écouter le son reconstruit ---
    # Vous pouvez aussi l'enregistrer dans un fichier wav si vous préférez
    # wavfile.write('guitare_filtree.wav', fe, son_filtre.astype(np.int16))
    sd.play(son_filtre, fe)
    status = sd.wait()  # Attendre que le son soit joué



