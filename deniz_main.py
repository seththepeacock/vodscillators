from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from plots import *
import seaborn as sns


if 1==1:
    # Open pickled vodscillator
    with open("vd-no-noise.pkl", 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)

if 0==1:
# Open pickled vodscillator
    with open("vd-w-noise.pkl", 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)

#v.plotter("cluster")
#v.plotter("PSD")
#v.plotter("superimpose")
#phase_portrait(v)

heat_map(v)
plt.show()

