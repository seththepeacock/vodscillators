from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from plots import *
import seaborn as sns


# Open pickled vodscillator
filename = "vd-w-noise.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)


#v.plotter("cluster")
#v.plotter("PSD")
#v.plotter("superimpose")

n = v.num_osc
spectra = (abs(v.every_fft))**2  #first index is oscillator index
avgd_spectra = np.squeeze(np.average(spectra, axis=1)).transpose() #avging over runs
osc_array = np.arange(0, n, 1)
freq_array = v.fft_freq


xx, yy = np.meshgrid(osc_array, freq_array) 

print(avgd_spectra.shape, xx.shape, yy.shape)

#sns.heatmap(avgd_spectra.transpose())
plt.pcolormesh(xx, yy, avgd_spectra)
plt.colorbar()
plt.show()

