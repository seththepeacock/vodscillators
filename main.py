from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle

# Open pickled vodscillator
filename = "Frank.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)

v.plot_spectra(xmin = -0.1, xmax = 1)
#v.plot_freq_clusters()




