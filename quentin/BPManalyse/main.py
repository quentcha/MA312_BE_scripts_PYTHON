import cv2
import numpy as np
import matplotlib.pyplot as plt
def autocorrelation(data):
    dataA=np.fft.rfft(data)
    dataB=dataA[::-1]
    dataC=dataA*dataB
    return np.fft.irfft(dataC)
def passe_coupe_bande(fmin,fmax,data,freq):
    spectre = np.fft.rfft(data)
    spectre_coupe= np.zeros(len(spectre), dtype=complex)
    for i in range(len(freq)):
        if freq[i]>fmin and freq[i]<fmax:
            spectre_coupe[i]=spectre[i]
    return np.fft.irfft(spectre_coupe)
def analyse(data):
    data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
    data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))]) #optimisation pour fft

    #print("AUTOCORRELATION")
    #data=autocorrelation(data)

    mini,maxi=35,180#plage  accepté
    print("FILTRAGE DE",mini/60,"Hz à",maxi/60,"Hz")
    fe = video.get(cv2.CAP_PROP_FPS)
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)
    data=passe_coupe_bande(mini/60,maxi/60,data,freq)#filtrage en fonction de la plage de bpm

    print("TRANSFORMATION DE FOURIER")
    spectre=np.fft.rfft(data)
    idx_max = np.argmax(abs(spectre))# Trouve l'indice du pic d'intensité
    #print("RESULTAT :",freq[idx_max]*60,"BPM")

    #print("PLOTTING")
    #plt.plot(freq,abs(spectre))
    #plt.xlabel("Frequence , Hz " )
    #plt.ylabel("Intensité")
    #plt.show()
    #print("QUITTEZ LE PLOT POUR REPRENDRE DES MESURES")

    return freq[idx_max]*60

#(°,°,°) -> (blue, green, red)
#eyecascade = cv2.CascadeClassifier('haarcascade_eye.xml')
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
nbr_capteurs = 10
longueur_captation=200
compteur = 0
som=0

data=np.empty([nbr_capteurs,longueur_captation])

print("start")
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            posX=np.linspace(x+(w//4),x+w - (w//4),nbr_capteurs)
            for x in range(len(posX)):
                pixel_value=frame[y + (h//5), x+int(posX[x])][1]#on récupère l'intensité du canal vert car plus efficace pour détecter les vaisseaux sanguins
                cv2.rectangle(frame, (x + int(posX[x])-2, y + (h//6)), (x + int(posX[x]), y + (h//6)+2), (0, 0, 255), 2)
                data[x,compteur]=pixel_value#abs(data[x][-1]-pixel_value))
            print(f"COLLECTE D'IMAGE :{round(compteur/200,2)*100} % | RESULTAT PRECEDENT : {som/nbr_capteurs} BPM")
            compteur+=1
    if len(faces)==0:
        compteur=0
        data=np.empty([nbr_capteurs,longueur_captation])
        print("CAMERA NE DETECTE RIEN")#pour éviter les hallucinatios et avoir un dataset fiable
    if (compteur)==longueur_captation:
        som=0
        for data_capteur in data:
            res=analyse(data_capteur)
            som+=res
            print(f"RESULTAT : {res} BPM")
        print(f"RESULTAT MOYEN: {som/nbr_capteurs} BPM")
        data=np.empty([nbr_capteurs,longueur_captation])
        compteur=0
        #input("press any key to resume")
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    # permet d’interrompre la vidéo par l’appui de la touche 'q' du clavier.
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

