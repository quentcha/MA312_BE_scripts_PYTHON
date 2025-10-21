import cv2
import numpy as np
#(°,°,°) -> (blue, green, red)
eyecascade = cv2.CascadeClassifier('haarcascade_eye.xml')
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)
compteur = 0

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
            eyes = eyecascade.detectMultiScale(visage_gray)
            for (ex, ey, ew, eh) in eyes:
                if ex > 0 and ey > 0:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
    #cv2.rectangle(frame, (x - ex, y - ey), (x - ex - ew, y - ey - eh), (0, 0, 255), 2)
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    # permet d’interrompre la vidéo par l’appui de la touche 'q' du clavier.
    if key == ord('q'):
        break

print(compteur)
video.release()
cv2.destroyAllWindows()

