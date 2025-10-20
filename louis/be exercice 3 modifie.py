import cv2
import numpy as np
from collections import deque

# --- FONCTION POUR RÉCUPÉRER L'INTENSITÉ DU FRONT ---
def get_forehead_intensity(frame, face_rect):
    """
    Calcule l'intensité moyenne de la couleur verte sur une zone du front.

    Args:
        frame: L'image complète en couleur (format BGR).
        face_rect: Le rectangle (x, y, w, h) délimitant le visage.

    Returns:
        L'intensité moyenne du canal vert sur la zone du front, ou None si la zone est invalide.
    """
    x, y, w, h = face_rect

    # Définition d'une zone approximative pour le front
    # On prend une zone au-dessus du centre horizontal et dans le quart supérieur du visage
    forehead_x = x + w // 8
    forehead_y = y + h // 16
    forehead_w = w // 4
    forehead_h = h // 8
    
    # S'assurer que la zone est valide
    if forehead_h > 0 and forehead_w > 0:
        # Découpage de la région d'intérêt (ROI) du front
        forehead_roi = frame[forehead_y : forehead_y + forehead_h, forehead_x : forehead_x + forehead_w]
        
        # Séparation des canaux de couleur (BGR)
        b, g, r = cv2.split(forehead_roi)
        
        # Calcul de la moyenne du canal vert (g)
        # np.mean est plus robuste car il gère les zones vides sans erreur
        return np.mean(g)
        
    return None

# --- INITIALISATION ---
# Chargement des classifieurs en cascade de Haar
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # Gardé pour le contexte, mais non utilisé pour l'intensité

# Initialisation de la capture vidéo
video = cv2.VideoCapture(0)

# Création du vecteur de pixels avec une longueur maximale fixe (ex: 100 dernières valeurs)
# deque est une structure de données optimisée pour ajouter/retirer des éléments aux extrémités
pixel_vector = deque(maxlen=100)

# --- BOUCLE PRINCIPALE ---
while True:
    check, frame = video.read()
    if not check:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # On ne traite que le plus grand visage détecté pour plus de stabilité
    if len(faces) > 0:
        # Trier les visages par leur surface (largeur * hauteur) et prendre le plus grand
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        main_face = faces[0]
        x, y, w, h = main_face

        # 1. Récupérer l'intensité du front
        intensity = get_forehead_intensity(frame, main_face)
        
        if intensity is not None:
            # 2. Rafraîchir le vecteur de pixels
            pixel_vector.append(intensity)
            # Affichage de l'intensité et de la taille du vecteur pour vérification
            print(f"Vecteur de pixels (taille: {len(pixel_vector)}): {list(pixel_vector)}")
            print(f"Intensité actuelle du front: {intensity:.2f}")

        # Dessin des rectangles pour la visualisation
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Visage en bleu
        # Visualiser la zone du front utilisée
        cv2.rectangle(frame, (x + w//4, y + h//8), (x + w//4 + w//2, y + h//8 + h//4), (0, 255, 255), 2) # Front en jaune

    # Affichage de la vidéo
    cv2.imshow("Webcam", frame)
    
    # Interruption avec la touche 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()