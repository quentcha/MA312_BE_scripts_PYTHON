from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import time
def SINC(amplitude,t,freq):
   # return amplitude*np.sin(2*np.pi*t*freq)
    return amplitude*np.sinc(np.pi*t*freq)
def bruit(amplitude,t,freq):
    return amplitude*np.random.randn(len(t))*np.exp(-50*t)
##hi hat
'''
duree=1
fe =44100
freq =440
amplitude =0.8
n=6

t=np.linspace(0 , duree , fe*duree) # ou la version avec arange : np.arange(0, duree ,1/fe ,dtype = np.float32)
#signal=np.zeros(len(t))
#for i in range(1,n):
signal=sum([SINC((1/i)*amplitude,t,i*freq) for i in range(1,30)])
signal+=sum([bruit((1/i)*amplitude,t,2*freq) for i in range(1,30)])
# f(amplitude,t,freq)+f((1/2)*amplitude,t,2*freq)+f((1/3)*amplitude,t,3*freq)+f((1/4)*amplitude,t,4*freq)+f((1/5)*amplitude,t,5*freq)
plt.plot(t, signal)
plt.show()

sd.play(signal, fe)
time.sleep(len(signal) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()
'''
##note
def note(amplitude,t,note):
    freq={}
    return amplitude*np.sin(2*np.pi*t*freq)
duree=1
fe =44100
freq =440
amplitude =0.8
n=6

t=np.linspace(0 , duree , fe*duree) # ou la version avec arange : np.arange(0, duree ,1/fe ,dtype = np.float32)
#signal=np.zeros(len(t))
#for i in range(1,n):
signal=f(amplitude,t,freq)+f((1/2)*amplitude,t,2*freq)+f((1/3)*amplitude,t,3*freq)+f((1/4)*amplitude,t,4*freq)+f((1/5)*amplitude,t,5*freq)
plt.plot(t, signal)
plt.show()

sd.play(signal, fe)
time.sleep(len(signal) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()
