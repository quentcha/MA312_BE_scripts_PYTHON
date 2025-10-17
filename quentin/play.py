import numpy as np
import sounddevice as sd
import time

def sound(data,fe):
    data_norm = data / (np.max(np.abs(data)) + 1e-12)
    sd.play(data_norm, fe)
    time.sleep(len(data_norm) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
    sd.stop()
