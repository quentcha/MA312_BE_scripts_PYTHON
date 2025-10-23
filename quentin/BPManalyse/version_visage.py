import cv2
import numpy as np
import matplotlib.pyplot as plt
def autocorrelation(data):
    n=len(data)
    correlated_data=np.zeros(n)
    for t in range(n):
        #print(f"AUTOCORRELATION : {round(t/n,2)*100} %")
        correlated_data[t]=sum(np.conj(data[:n-t])*data[t:])
    return correlated_data
def passe_coupe_bande(fmin,fmax,data,freq):
    spectre = np.fft.rfft(data)
    spectre_coupe= np.zeros(len(spectre), dtype=complex)
    for i in range(len(freq)):
        if freq[i]>fmin and freq[i]<fmax:
            spectre_coupe[i]=spectre[i]
    return np.fft.irfft(spectre_coupe)
def analyse(data):
    data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))]) #optimisation pour fft

    mini,maxi=35,180#plage accepté
    print("FILTRAGE DE",mini/60,"Hz à",maxi/60,"Hz")
    fe = 30
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)
    data=passe_coupe_bande(mini/60,maxi/60,data,freq)#filtrage en fonction de la plage de bpm

    print("AUTOCORRELATION")
    data=autocorrelation(data)

    print("TRANSFORMATION DE FOURIER ------------------------------------------------------------------------")
    spectre=np.fft.rfft(data)
    max = np.argmax(abs(spectre))# Trouve l'indice du pic d'intensité

    top=[]
    copySpectre=np.copy(spectre)
    for i in range(10):
        copySpectre=np.copy(copySpectre)
        id=np.argmax(abs(copySpectre))
        top.append(freq[id]*60)
        copySpectre[id]=0

    return top,data

#(°,°,°) -> (blue, green, red)
print("CLASSIFICATION HAARCASCADE VISAGE")
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("INITIALISATION CAMERA")
video = cv2.VideoCapture(0)
print("INITIALISATION VARIABLES")
nbr_capteurs = 10
longueur_captation=500
compteur = 0
res=0

FaceData=np.zeros(longueur_captation)
ConData=np.zeros(longueur_captation)

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            FaceX=np.linspace(x + (w // 4), x + w - (w // 4), nbr_capteurs)
            FacePixel=np.zeros(nbr_capteurs)
            for x in range(len(FaceX)):
                FacePixel_value=frame[y + (h//8), x+int(FaceX[x])][1]#on récupère l'intensité du canal vert car plus efficace pour détecter les vaisseaux sanguins
                cv2.rectangle(frame, (x + int(FaceX[x])-2, y + (h//8)), (x + int(FaceX[x]), y + (h//8)+2), (0, 0, 255), 2)
                FacePixel[x]=FacePixel_value
            FaceData[compteur]=abs(FaceData[compteur-1]-np.mean(FacePixel))

            ConPixel_value=frame[1, 1][1]
            cv2.rectangle(frame, (0, 0), (2, 2), (0, 0, 255), 2)
            ConData[compteur]=ConPixel_value

            print(f"COLLECTE D'IMAGE :{round(compteur/longueur_captation,2)*100} % | RESULTAT PRECEDENT : {res} BPM")
            compteur+=1

    if len(faces)==0:
        compteur=0
        FaceData=np.zeros(longueur_captation)
        ConData=np.zeros(longueur_captation)
        print("CAMERA NE DETECTE RIEN")#pour éviter les hallucinations et avoir un dataset fiable

    if (compteur)==longueur_captation:
        FaceData /= (np.max(np.abs(FaceData)) + 1*10**(-12))#normalisation
        ConData /= (np.max(np.abs(FaceData)) + 1*10**(-12))#normalisation
        topFace,dataFace=analyse(FaceData)
        print(f"PIQUES RESULTAT : {topFace}")
        topControl,dataControl=analyse(ConData)
        print(f"PIQUES RESULTAT CONTROLE: {topControl}")
        res=0
        for val in topFace:
            if val not in topControl and 35<val<180:
                res=val
                break
        if res!=0 : print(f"RESULTAT : {res} BPM")
        else: print(f"TROP DE BRUIT LUMINEUX POUR CAPTER LE BPM, ESSAYEZ DE CHANGER D'ENVIRONNEMENT")

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

