from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd
import time
def SINC(amplitude,t,freq):
   # return amplitude*np.sin(2*np.pi*t*freq)
    return amplitude*np.sinc(np.pi*t*freq)
def bruit_sec(amplitude,t):
    return amplitude*np.random.randn(len(t))*np.exp(-50*t)
def hihat(amplitude,t, freq=440,nbr_harmo=30):
    sig= sum([SINC((1/i)*amplitude,t,i*freq) for i in range(1,nbr_harmo)])
    sig+=sum([bruit_sec((1/i)*amplitude,t) for i in range(1,nbr_harmo)])
    return sig
def SIN(amplitude,t,freq):
    return amplitude*np.sin(2*np.pi*t*freq)#*np.exp((-1.1/t))# attack
def note(amplitude,t,note,nbr_harmo=30):
    NoteToFreq= {
    'do-1': 16.35, 'do0': 32.70, 'do1': 65.41, 'do2': 130.81, 'do3': 261.63, 'do4': 523.25, 'do5': 1046.50, 'do6': 2093.00, 'do7': 4186.01, 'do8': 8372.02, 'do9': 16744.04,
    'do#-1': 17.33, 'do#0': 34.65, 'do#1': 69.30, 'do#2': 138.59, 'do#3': 277.18, 'do#4': 554.37, 'do#5': 1108.73, 'do#6': 2217.46, 'do#7': 4434.92, 'do#8': 8869.84, 'do#9': 17739.68,
    'ré-1': 18.36, 'ré0': 36.71, 'ré1': 73.42, 'ré2': 146.83, 'ré3': 293.66, 'ré4': 587.33, 'ré5': 1174.66, 'ré6': 2349.32, 'ré7': 4698.64, 'ré8': 9397.28, 'ré9': 18794.56,
    'ré#-1': 19.45, 'ré#0': 38.89, 'ré#1': 77.78, 'ré#2': 155.56, 'ré#3': 311.13, 'ré#4': 622.25, 'ré#5': 1244.51, 'ré#6': 2489.02, 'ré#7': 4978.03, 'ré#8': 9956.06, 'ré#9': 19912.12,
    'mi-1': 20.60, 'mi0': 41.20, 'mi1': 82.41, 'mi2': 164.81, 'mi3': 329.63, 'mi4': 659.26, 'mi5': 1318.51, 'mi6': 2637.02, 'mi7': 5274.04, 'mi8': 10548.08, 'mi9': 21096.16,
    'fa-1': 21.83, 'fa0': 43.65, 'fa1': 87.31, 'fa2': 174.61, 'fa3': 349.23, 'fa4': 698.46, 'fa5': 1396.91, 'fa6': 2793.83, 'fa7': 5587.65, 'fa8': 11175.30, 'fa9': 22350.60,
    'fa#-1': 23.13, 'fa#0': 46.25, 'fa#1': 92.50, 'fa#2': 185.00, 'fa#3': 369.99, 'fa#4': 739.99, 'fa#5': 1479.98, 'fa#6': 2959.96, 'fa#7': 5919.91, 'fa#8': 11839.82, 'fa#9': 23679.64,
    'sol-1': 24.50, 'sol0': 49.00, 'sol1': 98.00, 'sol2': 196.00, 'sol3': 392.00, 'sol4': 783.99, 'sol5': 1567.98, 'sol6': 3135.96, 'sol7': 6271.93, 'sol8': 12543.86, 'sol9': 25087.72,
    'sol#-1': 25.96, 'sol#0': 51.91, 'sol#1': 103.83, 'sol#2': 207.65, 'sol#3': 415.30, 'sol#4': 830.61, 'sol#5': 1661.22, 'sol#6': 3322.44, 'sol#7': 6644.88, 'sol#8': 13289.76, 'sol#9': 26579.52,
    'la-1': 27.50, 'la0': 55.00, 'la1': 110.00, 'la2': 220.00, 'la3': 440.00, 'la4': 880.00, 'la5': 1760.00, 'la6': 3520.00, 'la7': 7040.00, 'la8': 14080.00, 'la9': 28160.00,
    'la#-1': 29.14, 'la#0': 58.27, 'la#1': 116.54, 'la#2': 233.08, 'la#3': 466.16, 'la#4': 932.33, 'la#5': 1864.66, 'la#6': 3729.31, 'la#7': 7458.62, 'la#8': 14917.24, 'la#9': 29834.48,
    'si-1': 30.87, 'si0': 61.74, 'si1': 123.47, 'si2': 246.94, 'si3': 493.88, 'si4': 987.77, 'si5': 1975.53, 'si6': 3951.07, 'si7': 7902.13, 'si8': 15804.26, 'si9': 31608.52
    }
    return sum([SIN((1/i)*amplitude,t,i*NoteToFreq[note]) for i in range(1,nbr_harmo)])
def bruit(amplitude,t):
    return amplitude*np.random.randn(len(t))
def BASS(amplitude,t,freq=50,nbr_harmo=30):
    a= sum([SIN((1/i)*amplitude,t,i*freq) for i in range(1,nbr_harmo)])
    #a+=sum([bruit(amplitude/100,t) for i in range(1,nbr_harmo)])
    a*=SIN(amplitude,t,np.pi)
    return a
##hi hat
'''
duree=1
fe =44100
amplitude =0.8
n=6

t=np.linspace(0 , duree , fe*duree) # ou la version avec arange : np.arange(0, duree ,1/fe ,dtype = np.float32)
#signal=np.zeros(len(t))
#for i in range(1,n):
signal=hihat(amplitude, t,30)
# f(amplitude,t,freq)+f((1/2)*amplitude,t,2*freq)+f((1/3)*amplitude,t,3*freq)+f((1/4)*amplitude,t,4*freq)+f((1/5)*amplitude,t,5*freq)
plt.plot(t, signal)
plt.show()

sd.play(signal, fe)
time.sleep(len(signal) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()
'''

##note
duree=2
fe =44100
amplitude =0.8
n=6

t_piano=np.linspace(0 , duree , fe*duree) # ou la version avec arange : np.arange(0, duree ,1/fe ,dtype = np.float32)
piano=np.block([note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30)+note(amplitude,t_piano,'sol3',30),
                note(amplitude,t_piano,'si2',30)+note(amplitude,t_piano,'ré3',30)+note(amplitude,t_piano,'fa3',30),
                 note(amplitude,t_piano,'la2',30)+note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30),
                 note(amplitude,t_piano,'la2',30)+note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30)+note(amplitude*5,t_piano,'do0',30),
                 note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30)+note(amplitude,t_piano,'sol3',30),
                 note(amplitude,t_piano,'la2',30)+note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30)+note(amplitude*5,t_piano,'do0',30),
                note(amplitude,t_piano,'si2',30)+note(amplitude,t_piano,'ré3',30)+note(amplitude,t_piano,'fa3',30),
                note(amplitude,t_piano,'la2',30)+note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30),
                note(amplitude,t_piano,'la2',30)+note(amplitude,t_piano,'do3',30)+note(amplitude,t_piano,'mi3',30)])

t_percu=np.linspace(0 , duree, (fe*duree//2))
percu=np.block([hihat(20*amplitude, t_percu,30)+note(amplitude,t_percu,['sol2','mi2','fa2','sol2'][i%4],30) for i in range(2*9)])

t_percu2=np.linspace(0 , duree, (fe*duree//4))
percu2=np.block([hihat(10*amplitude, t_percu2,30) for i in range(4*9)])

t_percu3=np.linspace(0 , duree, (fe*duree//3))
percu3=np.block([hihat(10*amplitude, t_percu3,30) for i in range(3*9)])

signal=piano+percu+percu2+percu3
signal=np.block([signal for i in range(10)])
#plt.plot(signal)
#plt.show()
sd.play(signal, fe)
time.sleep(len(signal) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()

'''
### Bass
duree=20
fe =44100
amplitude =0.8
n=6

t_piano=np.linspace(0 , duree//8 , (fe*duree//8)) # ou la version avec arange : np.arange(0, duree ,1/fe ,dtype = np.float32)
piano=np.block([BASS(amplitude,t_piano,261.63,30)+BASS(amplitude,t_piano,329.63,30)+BASS(amplitude,t_piano,392.00,30),
                BASS(amplitude,t_piano,246.94,30)+BASS(amplitude,t_piano,293.66,30)+BASS(amplitude,t_piano,349.23,30),
                 BASS(amplitude,t_piano,220.00,30)+BASS(amplitude,t_piano,261.63,30)+BASS(amplitude,t_piano,329.63,30),
                 BASS(amplitude,t_piano,261.63,30)+BASS(amplitude,t_piano,329.63,30)+BASS(amplitude,t_piano,392.00,30),
                 BASS(amplitude,t_piano,220.00,30)+BASS(amplitude,t_piano,261.63,30)+BASS(amplitude,t_piano,329.63,30),
                BASS(amplitude,t_piano,246.94,30)+BASS(amplitude,t_piano,293.66,30)+BASS(amplitude,t_piano,349.23,30),
                BASS(amplitude,t_piano,220.00,30)+BASS(amplitude,t_piano,261.63,30)+BASS(amplitude,t_piano,329.63,30)])
signal=piano
#plt.plot(signal)
#plt.show()
print("playing")
sd.play(signal, fe)
time.sleep(len(signal) / fe)#permet au programme d’attendre la fin de la lecture du son avant de se terminer
sd.stop()

'''
