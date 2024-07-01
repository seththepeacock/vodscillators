from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle

filename = "tiny_frank.pkl"

with open(filename, 'rb') as picklefile:
    v = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, and thus display its documentaiton for you!
    assert isinstance(v, Vodscillator)

# Now we can use our vodscillator v with its solution pre-solved for!
# We can get the solution (complex output as a function of time) with v.sol[index] where "index" is the index of the oscillator. 
# If we want the summed solution (all of the oscillators summed together) we grab v.summed_sol 


""" x = v.fft_freq
y = np.abs(v.every_fft[1][0])

plt.plot(x, y)
plt.show()
 """
# Generating V&D figs 2-5 style clustering graphs

#charfreq = v.omegas / (2*np.pi)

v.get_fft()
