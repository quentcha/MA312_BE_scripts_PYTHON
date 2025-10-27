import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def egalisateur(data, k1, k2, k3, k4, k5, k6):  # Soient ki, les coefficients multiplicateurs des bandes définies

    fe = 44100
    spectre = np.fft.rfft(data)  # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(data), d=1.0 / fe)

    # On définit les indices des 6 plages de fréquence:
    b_index_min = np.searchsorted(freq, 0)  # Basse
    b_index_max = np.searchsorted(freq, 100)

    bm_index_min = np.searchsorted(freq, 101)  # Basse médium
    bm_index_max = np.searchsorted(freq, 200)

    mg_index_min = np.searchsorted(freq, 201)  # Médium grave
    mg_index_max = np.searchsorted(freq, 400)

    ma_index_min = np.searchsorted(freq, 401)  # Médium aigu
    ma_index_max = np.searchsorted(freq, 800)

    a_index_min = np.searchsorted(freq, 801)  # Aigu
    a_index_max = np.searchsorted(freq, 5000)

    ta_index_min = np.searchsorted(freq, 5000)  # Très aigu
    ta_index_max = np.searchsorted(freq, 20000)

    # On applique les facteurs à leur plage de fréquences:

    spectre_filtre = spectre.copy()

    spectre_filtre[b_index_min:b_index_max] = k1 * spectre_filtre[b_index_min:b_index_max]
    spectre_filtre[bm_index_min:bm_index_max] = k2 * spectre_filtre[bm_index_min:bm_index_max]
    spectre_filtre[mg_index_min:mg_index_max] = k3 * spectre_filtre[mg_index_min:mg_index_max]
    spectre_filtre[ma_index_min:ma_index_max] = k4 * spectre_filtre[ma_index_min:ma_index_max]
    spectre_filtre[a_index_min:a_index_max] = k5 * spectre_filtre[a_index_min:a_index_max]
    spectre_filtre[ta_index_min:ta_index_max] = k6 * spectre_filtre[ta_index_min:ta_index_max]

    # On renvoie le spectre initial et le spectre égalisé afin de les comparer:
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(freq, np.abs(spectre), "g")
    plt.plot(freq, np.abs(spectre_filtre), "r")
    plt.show()

    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:
    son = np.fft.irfft(spectre_filtre)
    '''son = np.fft.irfft(spectre_filtre)
    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    sd.play(x,fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()'''

    return son

def filtrage_amp(data, A1, A2):  # Soient A1 et A2 les amplitudes des bornes

    fe, x = wavfile.read(data)
    x = x.astype(np.float32)
    if x.ndim == 2:     # On extrait le son
        x = x.mean(axis=1)
    x /= (np.max(np.abs(x)) + 1e-12)

    spectre = np.fft.rfft(x)  # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(x), d=1.0 / fe)

    spectre_filtre = np.zeros(len(spectre), dtype=complex)  # On crée une liste vide qui va accueillir le spectre filtré
    for i in range(len(spectre)):
        a = np.abs(spectre[i])
        if a < A1 or a > A2:
            spectre_filtre[i] = 0
        else:
            spectre_filtre[i] = a

    # On renvoie le plot du spectre initial et du spectre filtré pour les comparer:

    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(freq, np.abs(spectre), "r")
    plt.plot(freq, np.abs(spectre_filtre), "g")
    plt.grid(True)
    plt.show()

    son = np.fft.irfft(spectre_filtre)
    return son

def seuillage(data, fe, thau, k):  # Soit thau le seuil et k le coefficient de réduction

    spectre = np.fft.rfft(data)  # On calcule le spectre du signal avec la transformée de Fourier
    freq = np.fft.rfftfreq(len(data), d=1.0 / fe)
    spectre_filtre = []  # On crée une liste vide qui va accueillir le spectre filtré
    for i in range(len(spectre)):
        a = np.abs(spectre[i])
        if a < thau:
            spectre_filtre.append(k * a)
        else:
            spectre_filtre.append(a)

    son = np.fft.irfft(spectre_filtre)

    # On renvoie le plot du spectre initial et du spectre filtré pour les comparer:

    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(freq, np.abs(spectre), "r")
    plt.plot(freq, np.abs(spectre_filtre), "g")
    plt.show()

    '''
    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:


    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    #sd.play(x,fe)
    #time.sleep(len(son) / fe)  # permet d'écouter un son
    #sd.stop()'''
    return son

def tremolo(signal, sampling_rate= 44100, ft=0.5, depth=1):

    t = np.linspace(0, len(signal)/sampling_rate, len(signal))
    son = signal * (1 - depth * np.abs(np.cos(np.pi * t *ft)))
    return son

def ring_modulation(signal, sampling_rate= 44100, fp=400.0):

    t = np.linspace(0, len(signal) / sampling_rate, len(signal))

    # On crée l'onde porteuse que nous allons appliquer lors de la modulation
    carrier_wave = np.sin(2 * np.pi * fp * t)

    # On appliquee la modulation en anneau
    signal_traite = signal * carrier_wave

    # On normalise pour éviter la saturation après la multiplication
    max_val = np.max(np.abs(signal_traite))
    if max_val > 0:
        signal_traite /= max_val

    return signal_traite

def frequency_shift(data,shift,fe):#erreur va s'additionner car on arrondi les perte des hautes fréquences

    spectre = np.fft.rfft(data)
    freq=np.fft.rfftfreq(data.size, d=1./fe)
    pas=freq[1]-freq[0]
    if shift>0:
        spectre = np.block([np.zeros(int(abs(shift)/pas)),spectre])
    elif shift<0:
        spectre = spectre[int(abs(shift)/pas):]
    return np.fft.irfft(spectre,n=len(data))

def autocorrelation(data):# L'autocorrelation prend en entrée les données temporelles
    # On applique la tranformée de Fourier sur les données temporelles
    dataA=np.fft.rfft(data)
    #On applique la transformée de Fourier sur le conjugué des données temporelles
    dataB=np.fft.rfft(np.conj(data))
    #On multiplie la transformée des données temporelles avec la transformée du conjugué des données temporelles
    correlated_data=dataA*dataB
    #On applique la tranformée de Fourier inverse
    return np.fft.irfft(correlated_data)

def passe_bande(fmin,fmax,data,freq):
    # La fonction prend en entrée :
    # la fréquence minimale et maximale souhaitée
    # l'array d'intensitées associé à chaque indice : data
    # l'array de frequences associées à chaque indice : freq
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

def coupe_bande(fmin,fmax,data,freq):
    # La fonction prend en entrée :
    # la plage de fréquences que l'on souhaite enlevé
    # l'array d'intensités associés à chaque indice : data
    # l'array de frequences associées à chaque indice : freq
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




