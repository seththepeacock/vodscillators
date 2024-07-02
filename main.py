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


v.plotter(plot_type=['superimpose'], xmax = 10, ymax = 20)
plt.show()
