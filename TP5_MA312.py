from scipy.io import wavfile
import numpy as np
fe, x= wavfile.read('votreson.wav')
x = x.astype(np.float32)
if x.ndim == 2 : #stéréo -> mono si besoin
x = x.mean(axis =1)
x /= (np.max(np.abs(x))+1*10**(-12))#normalisation
