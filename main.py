from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle

# Open pickled vodscillator
filename = "vd-no-noise.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)


v.plotter("cluster")
v.plotter("PSD")
v.plotter("superimpose")
v.plot_waveform()



"""
logaritmic plot for coherence & psd
------------------------
v.psd()
v.coherence()
plt.plot(v.fft_freq, np.log(v.psd), label="psd")
plt.plot(v.fft_freq, v.SOO_phase_coherence*10, label="coherence")
plt.legend()
"""

#write stuff like this!!!
#vlodder.coherence(v.SOO_fft)

plt.show()

