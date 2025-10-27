import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))]) #optimisation pour fft
    #Fréquence d'échantillonage constante (référence caméra)
    fe = 30
    #Array de fréquences
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)
    #plage de valeurs acceptée
    mini,maxi=35,180
    print("FILTRAGE DE",mini/60,"Hz à",maxi/60,"Hz")
    data=passe_bande(mini/60,maxi/60,data,freq)#filtrage en fonction de la plage de bpm
    # On créer une liste de la fréquence et ses harmoniques associées au secteur
    freqSecteur=[i*50 for i in range(1,5)]
    print("FILTRAGE DE",freqSecteur[0],"Hz et ses harmoniques")
    #On applique le filtre coupe bande sur toutes les fréquences de la liste
    for f in freqSecteur:
        data=coupe_bande(f,f,data,freq)

    print("AUTOCORRELATION")
    data=autocorrelation(data)

    print("TRANSFORMATION DE FOURIER ------------------------------------------------------------------------")
    spectre=np.fft.rfft(data)
    max = np.argmax(abs(spectre))# Trouve l'indice du pic d'intensité
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
        topFace=analyse(FaceData)
        #On récupère les pics dominants du pixel controle
        print(f"PICS RESULTAT : {topFace}")
        topControl=analyse(ConData)
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

