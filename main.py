from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle

# Open pickled vodscillator
filename = "nonoise.pkl"
with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(v, Vodscillator)

start = timeit.default_timer()

#v.coherence()
#v.save()

stop = timeit.default_timer() 
print('Total time:', stop - start, "seconds, or", (stop-start)/60, "minutes") 

plt.plot(v.fft_freq, v.SOO_phase_coherence)
plt.show()
