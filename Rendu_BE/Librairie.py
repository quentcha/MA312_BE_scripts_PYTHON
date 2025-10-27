import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


def egalisateur(data, k1, k2, k3, k4, k5, k6):  # Soient ki, les coefficients multiplicateurs des bandes définies

    """
    Cette fonction permet d'amplifier ou de réduire l'amplitude de fréquences ciblées sur un signal sonore.
    On définit les 6 plages de fréquences auxquelles on appliquera les coefficients pris en compte dans la fonction:

    Basse : 0-100 Hz

    Basse-médium : 101-200 Hz

    Médium-grave : 201-400 Hz

    Médium-aigu : 401-800 Hz

    Aigu : 0,801-5 kHz

    Très aigu: 5-20 kHz"""

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

    """"Cette fonction renvoie un son débarrassé des fréquences où l'amplitude n'est pas comprise entre A1 et A2. Elle prend en argument le nom du fichier sonore que nous souhaitons filtrer"""

    fe, x = wavfile.read(data)
    x = x.astype(np.float32)
    if x.ndim == 2:
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
    '''
    # On applique la transformée inverse de Fourier pour récupérer le son égalisé:


    sd.play(son, fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()

    # Optionnel: on peut jouer à la suite les deux sons pour les comparer.
    sd.play(x,fe)
    time.sleep(len(son) / fe)  # permet d'écouter un son
    sd.stop()
    '''
    return son

def seuillage(data, fe, thau, k):  # Soit thau le seuil et k le coefficient de réduction

    '''
    Cette fonction a pour but de multiplier par un coefficient k (inférieur ou égal à 1) les amplitudes inférieures à thau. Ce procédé vise
    à diminuer les bruits de fond. Elle prend en argument l'array du fichier sonore.
    '''

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
    """
    Applique un effet de trémolo sur le signal.
    Le trémolo est un effet sonore qui applique une fonction sinusoïdale
    strictement positive (car une amplitude ne peut pas être négative) à l'amplitude du signal d'entrée.
    Cela aura pour effet de diminuer puis rétablir l'instensité du son de façon périodique.

    :param signal: Le signal audio d'entrée en array
    :param sampling_rate: La fréquence d'échantillonnage (Hz)
    :param ft (fréquence du trémolo): Détermine la périodicité de l'effet, généralement on prend des fréquences très basses afin que l'effet soit prononcé ( entre 0.5 et 10 Hz)
    :param depth: détermine l'amplitude de l'effet
    """

    t = np.linspace(0, len(signal)/sampling_rate, len(signal))
    son = signal * (1 - depth * np.abs(np.cos(np.pi * t *ft)))
    return son

def ring_modulation(signal, sampling_rate= 44100, fp=400.0):
    """
    Applique l'effet Ring Modulation.

    Le signal d'entrée est multiplié par une onde porteuse sinusoïdale
    à la fréquence.

    :param signal: Le signal audio d'entrée en array
    :param sampling_rate: Le taux d'échantillonnage (Hz)
    :param fp: Fréquence de la porteuse (Hz, souvent audible)
    :return: Signal avec l'effet de modulation en anneau appliqué
    """


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

def passe_bande(fmin,fmax,data,freq):
    """Cette fonction va supprimer sur notre bande de fréquence les fréquences se trouvant en dehors de fmin et fmax"""
    spectre = np.fft.rfft(data)
    spectre_coupe= np.zeros(len(spectre), dtype=complex) #L'array filtré contient initialement que des zeros
    for i in range(len(freq)):
        if freq[i]>fmin and freq[i]<fmax: #Si notre fréquence se trouve dans l'intervalle désiré alors on l'ajoute à l'array filtré
            spectre_coupe[i]=spectre[i]
    return np.fft.irfft(spectre_coupe)

def pitch(data,shift,fe):##erreur va s'additionner car on arrondi et perte des hautes fréquences
    """Ce code décale la fréquence de la fréquence shift"""
    spectre = np.fft.rfft(data)
    freq=np.fft.rfftfreq(data.size, d=1./fe)
    pas=freq[1]-freq[0]
    if shift>0: spectre = np.block([np.zeros(int(abs(shift)/pas)),spectre])
    elif shift<0: spectre = spectre[int(abs(shift)/pas):]
    return np.fft.irfft(spectre,n=len(data))




