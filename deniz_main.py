from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from vlodder import *

# Open pickled vodscillator
filename = "vd-w-noise.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)


#v.plotter("cluster")
#v.plotter("PSD")
#.plotter("superimpose")


heat_map(v)
    


plt.show()

