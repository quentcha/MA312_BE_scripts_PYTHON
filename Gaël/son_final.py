import sounddevice as sd
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from RingModulation import ring_modulation
from Tremolo import tremolo

fe, x = wavfile.read("sample1.wav")
x = x.astype(np.float32)
if x.ndim == 2:
    x = x.mean(axis=1)
x /= (np.max(np.abs(x)) + 1e-12)

tempo = 1/8
amplitude = 1
echantillonage = 44100
temps = tempo / 4
pause = np.linspace(0, temps, int(echantillonage * temps))
temps = tempo / 2
t = np.linspace(0, temps, int(echantillonage * temps))


sol = amplitude * np.sin(2 * np.pi * t * 392)
do =  amplitude * np.sin(2 * np.pi * t * 262)
rediez = amplitude * np.sin(2 * np.pi * t * 311.13)
soldiez = amplitude * np.sin(2 * np.pi * t * 415.30)
fa = amplitude * np.sin(2 * np.pi * t * 349.23)
re = amplitude * np.sin(2 * np.pi * t * 293.66)

signal = np.block([ring_modulation(sol) + ring_modulation(fa), ring_modulation(do) + ring_modulation(re), ring_modulation(pause), ring_modulation(rediez) + ring_modulation(do)])
signal = np.block([signal, signal, signal, tremolo(soldiez), tremolo(soldiez), tremolo(soldiez), tremolo(soldiez)])
signal = np.block([signal, signal])
sd.play(signal, echantillonage)
time.sleep(len(signal) / echantillonage)
sd.stop()
sd.stop()
