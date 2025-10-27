import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# L'autocorrelation prend en entrée les données temporelles
def autocorrelation(data):
    # On applique la tranformée de Fourier sur les données temporelles
    dataA=np.fft.rfft(data)
    #On applique la transformée de Fourier sur le conjugué des données temporelles
    dataB=np.fft.rfft(np.conj(data))
    #On multiplie la transformée des données temporelles avec la transformée du conjugué des données temporelles
    correlated_data=dataA*dataB
    #On applique la tranformée de Fourier inverse
    return np.fft.irfft(correlated_data)

# La fonction prend en entrée :
# la fréquence minimale et maximale souhaitée
# l'array d'intensitées associé à chaque indice : data
# l'array de frequences associées à chaque indice : freq
def passe_bande(fmin,fmax,data,freq):
    # On applique la transformée de Fourier discrète
    spectre = np.fft.rfft(data)
    # On initialise le spectre modifié
    spectre_coupe= np.zeros(len(spectre), dtype=complex)
    #Pour toute les fréquences, si la fréquence est comprise
    #dans la plage de valeur alors on la garde,
    #sinon elle est égale à 0
    for i in range(len(freq)):
        if freq[i]>=fmin and freq[i]<=fmax:
            spectre_coupe[i]=spectre[i]
        #On applique la transformée de Fourier inverse
    return np.fft.irfft(spectre_coupe)
# La fonction prend en entrée :
# la plage de fréquences que l'on souhaite enlevé
# l'array d'intensités associés à chaque indice : data
# l'array de frequences associées à chaque indice : freq
def coupe_bande(fmin,fmax,data,freq):
    #On applique la transformée de Fourier discrète
    spectre = np.fft.rfft(data)
    #On initialise le spectre modifié
    spectre_coupe= np.copy(spectre)
    #Pour toutes les fréquences, si la fréquence est comprise dans la plage de
    #valeur alors on la garde sinon elle est égale à 0
    for i in range(len(freq)):
        if freq[i]>=fmin and freq[i]<=fmax:
            spectre_coupe[i]=0
    # On applique la transformée de Fourier inverse
    return np.fft.irfft(spectre_coupe)
def analyse(data):
    # optimisation pour effectuer la transformée de Fourier rapide
    data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))])
    #Fréquence d'échantillonage constante (référence caméra)
    fe = 30
    #Array de fréquences
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)
    #plage de valeurs acceptée
    mini,maxi=35,180
    print("FILTRAGE DE",mini/60,"Hz à",maxi/60,"Hz")
    #Filtrage en fonction de la plage de BPM acceptables
    data=passe_bande(mini/60,maxi/60,data,freq)
    # On créer une liste de la fréquence et ses harmoniques associées au secteur
    freqSecteur=[i*50 for i in range(1,5)]
    print("FILTRAGE DE",freqSecteur[0],"Hz et ses harmoniques")
    #On applique le filtre coupe bande sur toutes les fréquences de la liste
    for f in freqSecteur:
        data=coupe_bande(f,f,data,freq)

    #On applique l'autocorrelation
    print("AUTOCORRELATION")
    data=autocorrelation(data)

    #On applique la transformée de Fourier pour passer du domaine temporel au domaine fréquentiel
    print("TRANSFORMEE DE FOURIER")
    spectre=np.fft.rfft(data)
    #Trouve l'indice du pic d'intensité dominant
    max = np.argmax(abs(spectre))
    #Initialisation d'une liste qui contiendra les fréquences dominantes
    top=[]
    #Initialisation d'un array facilement manipulable
    copySpectre=np.copy(spectre)
    #Recherche de l'index de l'intensité la plus importante
    id=np.argmax(abs(copySpectre))
    #On cherche les fréquences dominantes
    while abs(copySpectre[id])>1*10**(-10):
        #Ajout de la fréquence dominante à la liste top
        top.append(freq[id]*60)
        #Suppression de l'intensité liée à la fréquence dominante
        copySpectre[id]=0
        copySpectre=np.copy(copySpectre)
        #Detection de la fréquence dominante
        id=np.argmax(abs(copySpectre))

    return top

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
        maxFinger=analyse(FingerData)[0]
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
