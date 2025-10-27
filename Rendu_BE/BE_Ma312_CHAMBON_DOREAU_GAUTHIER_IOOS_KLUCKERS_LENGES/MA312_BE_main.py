#Author :
#Quentin CHAMBON
#Enzo DOREAU
#Gael GAUTHIER
#Axel IOOS
#Gaspar KLUCKERS
#Louis LENGES

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Egalisateur à 6 bandes
#On importe notre son test
fe, x = wavfile.read("Data\\20-20_000-Hz-Audio-Sweep.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)
fe = 44100
freq = np.fft.rfftfreq(len(x), d=1.0 / fe)

dataEgalise=lb.egalisateur(x, 1,1,1,1,2,2)  #On l'égalise pour faire ressortir les fréquences aigues (> 800 Hz)

# On renvoie le spectre initial et le spectre égalisé afin de les comparer:
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.plot(freq, np.abs(np.fft.rfft(dataEgalise)), "g")
plt.plot(freq, np.abs(np.fft.rfft(x)), "r")
plt.show()

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Filtrage ampliude
fe, x = wavfile.read("Data\guitare1.wav")
x = x.astype(np.float32)
if x.ndim == 2:     # On extrait le son
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)
freq = np.fft.rfftfreq(len(x), d=1.0 / fe)

data_filtre=lb.filtrage_amp(x, 2000, 8000)     #On garde sur le son de guitare les amplitudes entre 2000 et 8000

# On renvoie le plot du spectre initial et du spectre filtré pour les comparer:
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.plot(freq, np.abs(np.fft.rfft(x)), "r")
plt.plot(freq, np.abs(np.fft.rfft(data_filtre)), "g")
plt.grid(True)
plt.show()


#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Seuillage

fe, x = wavfile.read("Data\Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)
freq = np.fft.rfftfreq(len(x), d=1.0 / fe)

dataSeuillage=lb.seuillage(x, 10000, 0)

# On renvoie le plot du spectre initial et du spectre filtré pour les comparer:

plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.plot(freq, np.abs(np.fft.rfft(x)), "r")
plt.plot(freq, np.abs(np.fft.rfft(dataSeuillage)), "g")
plt.show()

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Tremolo
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

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Ring Modulation

fe, x = wavfile.read("Data\Série_de_Fourier_BE_Ma312_2025.wav")
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

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Passe Bande
fe, data= wavfile.read('Data\20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
freq = np.fft.rfftfreq(len(data), d=1.0/fe)

fmin,fmax=500,15000
plt.plot(freq,np.abs(np.fft.rfft(data)))
plt.plot(freq,np.abs(np.fft.rfft(lb.passe_bande(fmin,fmax,data,freq))))
plt.xlabel("Frequence , Hz ")
plt.ylabel("Amplitude")
plt.show()

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Frequency shift
fe, data= wavfile.read("Data\Série_de_Fourier_BE_Ma312_2025.wav")
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
#play.sound(data,fe)
data_shifted=lb.frequency_shift(data,500,fe)
freq_shifted=np.fft.rfftfreq(data_shifted.size, d=1./fe)
#play.sound(data_pitched,fe)
print("plotting...")
plt.plot(freq_shifted,np.abs(np.fft.rfft(data)))
plt.plot(freq_shifted,np.abs(np.fft.rfft(data_shifted)))

plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

plt.plot(data_shifted)
plt.show()

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Coupe Bande
fe, data= wavfile.read('Data\20-20_000-Hz-Audio-Sweep.wav')
data = data.astype(np.float32)
if data.ndim == 2 : #stéréo -> mono si besoin
    data = data.mean(axis =1)
data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
freq = np.fft.rfftfreq(len(data), d=1.0/fe)

fmin,fmax=5000,5000
plt.plot(freq,np.abs(np.fft.rfft(data)))
plt.plot(freq,np.abs(np.fft.rfft(lb.coupe_bande(fmin,fmax,data,freq))))
plt.xlabel("Frequence , Hz " )
plt.ylabel("Amplitude")
plt.show()

#%% TEST Son final
fe, x = wavfile.read("Data\Série_de_Fourier_BE_Ma312_2025.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

lenght_sec = 47998

extrait = x[5*lenght_sec:34*lenght_sec]
extrait1 = lb.tremolo(extrait[:7*lenght_sec], 44100, 0.035, 1)
extrait2 = lb.frequency_shift(lb.ring_modulation(lb.egalisateur(extrait[7*lenght_sec: int(12.4*lenght_sec)],1,1,2,4,4,4), 44100, 100), 100, 44100)
extrait3 = lb.frequency_shift(lb.ring_modulation(lb.egalisateur(extrait[int(12.4 *lenght_sec): int(17.95* lenght_sec) ],5,5,3,0,0,0), 44100, 100), 100, 44100)
extrait4 = lb.frequency_shift(lb.ring_modulation(lb.egalisateur(extrait[int(17.95*lenght_sec): 34*lenght_sec],2,2,2,4,4,4), 44100, 200), 200, 44100)
signal = np.block([ extrait1, extrait2, extrait3, extrait4])
sd.play(signal, fe)
time.sleep(len(extrait) / fe)  # permet d'écouter un son
sd.stop()

#%%
import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import MA312_BE_lib as lb

#TEST Pitch Shift
#Application du pitch
data_pitched = lb.pitch(data, 1)

# Calcul du spectre et de l’axe fréquentiel
freqs = np.fft.rfftfreq(len(data), d=1/fe)
fft_data = np.abs(np.fft.rfft(data))
fft_pitched = np.abs(np.fft.rfft(data_pitched))

#Tracé du spectre
plt.figure(figsize=(10,5))
plt.plot(freqs, fft_data, "g",  label="Original")
plt.plot(freqs, fft_pitched, "r", label="Décalé")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

#%%
#Detection du BPM a partir du front
import cv2
import numpy as np
#(°,°,°) -> (blue, green, red)
print("CLASSIFICATION HAARCASCADE VISAGE")
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("INITIALISATION CAMERA")
video = cv2.VideoCapture(0)
print("INITIALISATION VARIABLES")
#Nombre de pixels du front dont on souhaite avoir l'intensité
nbr_capteurs = 10
longueur_captation=500
compteur = 0
res=0
top_n=200

FaceData=np.zeros(longueur_captation)
ConData=np.zeros(longueur_captation)

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            #On créé un array avec les points du front en abscisse
            FaceX=np.linspace(x + (w//4), x + (3*w//4), nbr_capteurs)
            #On initialise un array qui contiendra les données pour chaque capteur
            FacePixel=np.zeros(nbr_capteurs)
            for pos in range(len(FaceX)):
                #Position en abscisse d'un capteur
                posx=int(FaceX[pos])
                FacePixel_value=frame[y + (h//8), posx][1]#on récupère l'intensité du canal vert car plus efficace pour détecter les vaisseaux sanguins
                cv2.rectangle(frame, (posx-2, y + (h//8)), (posx, y + (h//8)+2), (0, 0, 255), 2)
                FacePixel[pos]=FacePixel_value
                #On ajoute les données récoltées à l'array contenant les pixels du front
            FaceData[compteur]=np.mean(FacePixel)

            ConPixel_value=frame[1, 1][1]
            cv2.rectangle(frame, (0, 0), (2, 2), (0, 0, 255), 2)
            ConData[compteur]=ConPixel_value

            print(f"COLLECTE D'IMAGE :{round(compteur/longueur_captation,2)*100} % | RESULTAT PRECEDENT : {res}")
            compteur+=1
            break
    # Si aucun visage n'est détecté
    if len(faces)==0:
        #On initialise à nouveau nos variables
        compteur=0
        FaceData=np.zeros(longueur_captation)
        ConData=np.zeros(longueur_captation)
        print("CAMERA NE DETECTE RIEN")#pour éviter les hallucinations et avoir un dataset fiable

    if (compteur)==longueur_captation:
        #On normalise nos ensembles de données pour qu'ils soient comparables
        FaceData /= (np.max(np.abs(FaceData)) + 1*10**(-12))#normalisation
        ConData /= (np.max(np.abs(ConData)) + 1*10**(-12))#normalisation
        #On récupère les pics dominants des pixels du front
        topFace=lb.analyse(FaceData)
        #On récupère les pics dominants du pixel controle
        print(f"PICS RESULTAT : {topFace}")
        topControl=lb.analyse(ConData)
        print(f"PICS RESULTAT CONTROLE: {topControl}")
        #Si un résultat existe alors on le communique sinon on informe l'utilisateur qu'il y a trop de bruit
        res="TROP DE BRUIT POUR CAPTER LE BPM, ESSAYEZ DE CHANGER D'ENVIRONNEMENT"
        for val in topFace:
            #Si le pic dominant n'est pas présent dans les fréquences bruitées et s'il reste
            #dans le domaine des valeurs acceptables (car il arrive que les pics dominants soient
            #équivalents à 0 et en dehors des fréquences acceptables)
            if val not in topControl and 35<val<180:
                #Le résultat est donc ce pic dominant
                res=f"{val} BPM"
                break
        #On réinitialise nos variables afin de prendre de nouvelles mesures
        FaceData=np.zeros(longueur_captation)
        ConData=np.zeros(longueur_captation)
        compteur=0
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    # permet d’interrompre la vidéo par l’appui de la touche 'q' du clavier.
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

#%%
import cv2
import numpy as np
import time
#TEST Analyse du BPM a partir du doigt
#(°,°,°) -> (blue, green, red)
print("INITIALISATION CAMERA")
video = cv2.VideoCapture(0)
nbr_capteurs = 50
longueur_captation=250
compteur = 0
res=0

FingerData=np.zeros(longueur_captation)
print("PLACEZ VOTRE DOIGT SUR LA CAMERA POUR COMMENCER")
time.sleep(3)
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Xline=np.linspace(10, 630, nbr_capteurs)
    FingerPixel=np.zeros(nbr_capteurs)
    for x in range(len(Xline)):
        FingerPixel[x]=frame[250, int(Xline[x])][1]
        cv2.rectangle(frame, (int(Xline[x])-1,249), (int(Xline[x])+1, 251), (0, 0, 255), 2)
    FingerData[compteur]=np.mean(FingerPixel)

    print(f"COLLECTE D'IMAGE :{round(compteur/longueur_captation,2)*100} % | RESULTAT PRECEDENT : {res}")
    compteur+=1

    if (compteur)==longueur_captation:
        FingerData /= (np.max(np.abs(FingerData)) + 1*10**(-12))#normalisation
        maxFinger=lb.analyse(FingerData)[0]
        res=f"{maxFinger} BPM"
        print(f"RESULTAT : {res}")

        FingerData=np.zeros(longueur_captation)
        compteur=0
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    # permet d’interrompre la vidéo par l’appui de la touche 'q' du clavier.
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

