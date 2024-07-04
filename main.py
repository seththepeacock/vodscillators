from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle

# Open pickled vodscillator
filename = "vd-w-noise.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)


v.plotter("cluster")
v.plotter("PSD")
v.plotter("superimpose")



def c_psd_logplot(v): #logarithmic coherence and psd plot
    v.psd()
    v.coherence()
    plt.plot(v.fft_freq, np.log(v.psd), label="psd")
    plt.plot(v.fft_freq, v.SOO_phase_coherence*10, label="coherence")
    plt.legend()

def phase_portrait(v):
    xdot = np.imag((v.sol))
    x = np.real((v.sol))
    plt.plot(x, xdot)
    plt.grid()


#write stuff like this!!!
#vlodder.coherence(v.SOO_fft)

def heat_map(v):
    pass


plt.show()

