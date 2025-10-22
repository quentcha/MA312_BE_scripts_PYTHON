import cv2
import numpy as np
import matplotlib.pyplot as plt
def autocorrelation(data):
    dataA=np.fft.rfft(data)
    dataB=dataA[::-1]
    dataC=dataA*dataB
    return np.fft.irfft(dataC)
def passe_coupe_bande(fmin,fmax,spectre,freq):
    spectre_coupe= np.zeros(len(spectre), dtype=complex)
    for i in range(len(freq)):
        print(freq[i])
        if freq[i]>fmin and freq[i]<fmax:
            spectre_coupe[i]=spectre[i]
    return spectre_coupe
def analyse(data):
    print("TRANSFORMATION DE FOURIER")
    spectre=np.fft.rfft(data)
    #data /= (np.max(np.abs(data)) + 1*10**(-12))#normalisation
    #data = np.block([data, np.zeros(2**(int(np.log2(len(data)))+1)-len(data))]) optimisation pour fft
    mini,maxi=35,180#plage  accepté
    print("FILTRAGE DE",mini/60,"Hz à",maxi/60,"Hz")
    fe = video.get(cv2.CAP_PROP_FPS)
    freq = np.fft.rfftfreq(len(data), d=1.0/fe)
    data=passe_coupe_bande(mini/60,maxi/60,spectre,freq)#filtrage en fonction de la plage de bpm
    idx_max = np.argmax(abs(spectre))# Trouve l'indice du pic d'intensité
    print(idx_max)
    print("RESULTAT :",freq[idx_max],"Hz soit",freq[idx_max]*60,"BPM")
    #print(np.max(abs(spectre)))
    #freq=np.fft.rfftfreq(data.size, d=1./fe )
    print("PLOTTING")
    plt.plot(freq,abs(spectre))
    plt.xlabel("Frequence , Hz " )
    plt.ylabel("Intensité")
    plt.show()
    print("QUITTEZ LE PLOT POUR REPRENDRE DES MESURES")
#(°,°,°) -> (blue, green, red)
#eyecascade = cv2.CascadeClassifier('haarcascade_eye.xml')
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
compteur = 0
data=np.empty(1)
print("start")
while True:
    compteur = compteur + 1
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    l, c = gray.shape
    faces = facecascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            visage_gray = gray[y:y + h, x:x + w]
            '''
            eyes = eyecascade.detectMultiScale(visage_gray)
            for (ex, ey, ew, eh) in eyes:
                if ex > 0 and ey > 0:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            '''
            cv2.rectangle(frame, (x+ (3*w//4) - 2, y+10), (x +(3*w//4), y + 12), (0, 0, 255), 2)
            pixel_value=gray[y + 11, x + (3*w//4) - 1]#/255
    if len(faces)==0:
        data=np.empty(1)
        print("CAMERA NE DETECTE RIEN")
    else:
        data=np.append(data,pixel_value)
        print("COLLECTE D'IMAGE :",round(len(data)/200,2)*100,"%")
    if len(data)==200:
        analyse(data)
        data=np.empty(1)
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    # permet d’interrompre la vidéo par l’appui de la touche 'q' du clavier.
    if key == ord('q'):
        break

print(compteur)
video.release()
cv2.destroyAllWindows()

