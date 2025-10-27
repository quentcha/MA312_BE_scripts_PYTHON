import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def autocorrelation(data):
    dataA=np.fft.rfft(data)
    dataB=np.fft.rfft(np.conj(data))
    correlated_data=dataA*dataB
    return np.fft.irfft(correlated_data)
def passe_bande(fmin,fmax,data,freq):
    spectre = np.fft.rfft(data)
    spectre_coupe= np.zeros(len(spectre), dtype=complex)
    for i in range(len(freq)):
        if freq[i]>=fmin and freq[i]<=fmax:
            spectre_coupe[i]=spectre[i]
    return np.fft.irfft(spectre_coupe)
def coupe_bande(fmin,fmax,data,freq):
    spectre = np.fft.rfft(data)
    spectre_coupe= np.copy(spectre)
    for i in range(len(freq)):
        if freq[i]>=fmin and freq[i]<=fmax:
            spectre_coupe[i]=0
    return np.fft.irfft(spectre_coupe)
def analyse(data):
    data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))]) #optimisation pour fft
    fe = 30
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)

    freqSecteur=[i*50 for i in range(1,5)]
    print("FILTRAGE DE",freqSecteur[0],"Hz et ses harmoniques")
    for f in freqSecteur:
        data=coupe_bande(f,f,data,freq)#filtrage en fonction de la fréquence du secteur et ses harmoniques

    mini,maxi=35,180#plage accepté
    print("FILTRAGE DE",mini/60,"Hz à",maxi/60,"Hz")
    data=passe_bande(mini/60,maxi/60,data,freq)#filtrage en fonction de la plage de bpm

    print("AUTOCORRELATION")
    data=autocorrelation(data)

    print("TRANSFORMATION DE FOURIER ------------------------------------------------------------------------")
    spectre=np.fft.rfft(data)
    max = np.argmax(abs(spectre))# Trouve l'indice du pic d'intensité

    plt.stem(freq, abs(spectre), linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.xlabel("Frequence , Hz ")
    plt.ylabel("Intensité")
    plt.show()

    return freq[max]*60#conversion en BPM

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
        maxFinger=analyse(FingerData)
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
