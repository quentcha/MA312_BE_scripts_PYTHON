
import numpy as np
import matplotlib.pyplot as plt
'''
# %% pour la fonction cos
a = -10
b = 10
fe = 3
N = int((b - a) / (2 * np.pi) * fe)
kTe = np.linspace(a, b, N)
ech = np.cos(kTe)
Te = kTe[1] - kTe[0]

def sinc(x):
    if x == 0:
        return 1
    else:
        return np.sin(x) / x

vecsinc = np.vectorize(sinc)

def reconstru(t, ech, kTe, Te):
    l = len(t)
    s = np.zeros(l)
    indsomme = len(ech)
    for i in range(l):
        st = 0
        for k in range(indsomme):
            st += ech[k] * vecsinc(np.pi * (t[i] - kTe[k]) / Te)
        s[i] = st
    return s

t = np.linspace(a, b, 1000)
signal_recons = reconstru(t, ech, kTe, Te)

fig, ax = plt.subplots(2, figsize=(15, 7))
fig.suptitle("TD3 : Exercice 1 : théorème d'échantillonnage")
ax[0].set_title(f"Échantillonnage pour fe = {fe}")
ax[0].plot(t, np.cos(t), 'b', label="signal originel")
ax[0].plot(t, signal_recons, '-r', label="signal reconstruit")
ax[0].plot(kTe, ech, 'og', label="points échantillonnés")
ax[0].legend()

def TF(alpha):
    s = 10 * (vecsinc(10 * (1 - alpha)) + vecsinc(10 * (1 + alpha)))
    return s

alpha = np.linspace(-50, 50, 10000)
ax[1].set_title("Graphe de la transformée de Fourier")
ax[1].axvspan(-2 * np.pi * fe, 2 * np.pi * fe, alpha=0.5, color='red', label="plage des fréquences conservées")
ax[1].plot(alpha, TF(alpha), 'g')
ax[1].legend()

plt.show()
'''
fe=44100
freq=392
duree=1
t=np.linspace(0,duree,fe*duree)
amplitude=1
la=amplitude*np.sin(2*np.pi*t*freq)
import sounddevice as sd
sd.play (la, fe)
sd.wait()
