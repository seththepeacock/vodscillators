from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle

# Open pickled vodscillator
filename = "V+D fig 2A.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)

# start = timeit.default_timer()
# stop = timeit.default_timer() 
# print('Total time:', stop - start, "seconds, or", (stop-start)/60, "minutes") 

v.plot_freq_clusters()
