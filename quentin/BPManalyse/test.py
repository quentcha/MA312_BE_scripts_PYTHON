import cv2
import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(data):
    n=len(data)
    correlated_data=np.zeros(n)
    for t in range(n):
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

    print("AUTO-CORRELATION")
    data=autocorrelation(data)

    print("TRANSFORMATION DE FOURIER")
    spectre=np.fft.rfft(data)
    freq=np.fft.rfftfreq(data.size, d=1./30)

    max = np.argmax(abs(spectre))# Trouve l'indice du pic d'intensité
    max=freq[max]*60

    top=[]
    copySpectre=np.copy(spectre)
    for i in range(10):
        copySpectre=np.copy(copySpectre)
        id=np.argmax(abs(copySpectre))
        top.append(freq[id]*60)
        copySpectre[id]=0
    print(f"TOP RESULTAT : {top}")
    print(f"PICS RESULTAT : {max}")
    plt.stem(freq, abs(spectre), linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.xlabel("Frequence , Hz ")
    plt.ylabel("Intensité")
    plt.show()

#(°,°,°) -> (blue, green, red)
print("CLASSIFICATION HAARCASCADE VISAGE")
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("INITIALISATION CAMERA")
video = cv2.VideoCapture(0)
print("INITIALISATION VARIABLES")
longueur_captation=500
compteur = 0

FaceData=np.zeros(longueur_captation)
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            FaceData[compteur]=frame[y+11, x + (3*w//4)][1]
            cv2.rectangle(frame, (x+ (3*w//4) -1, y+10), (x + (3*w//4) - 1 , y + 12), (255, 0, 0), 2)
            compteur+=1

    if (compteur)==longueur_captation:
        res=analyse(FaceData)
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
