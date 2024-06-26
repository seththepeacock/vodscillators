from vodscillators import *
import matplotlib.pyplot as plt
import timeit
import pickle

with open("frank.pkl", 'rb') as picklefile:
    v = pickle.load(picklefile)
# Now we can use our vodscillator v with its solution pre-solved for!
# We can get the solution (complex output as a function of time) with v.sol[index] where "index" is the index of the oscillator. 
# If we want the summed solution (all of the oscillators summed together) we grab v.summed_sol 


# Generating V&D figs 2-5 style clustering graphs

#charfreq = v.omegas / (2*np.pi)


#main thing






