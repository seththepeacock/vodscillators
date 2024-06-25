from vodscillators import *
import matplotlib.pyplot as plt
import timeit
from scipy.integrate import solve_ivp
from numpy.fft import rfft, rfftfreq
from statistics import mean


start = timeit.default_timer() # starts timer that tells you code run time

params = {"name" : "Frank",
          "IC_method" : "rand",
          "freq_dist" : "exp",
          "loc_noise_amp" : 0.1,
          "glob_noise_amp" : 0.1,
          "epsilon" : 1.0,
          "omega_o" : 2*np.pi,
          "omega_n" : 5*np.pi,
          "d_R" : 0.15,
          "d_I" : -1.0,
          "B" : 1.0,
          "num_osc" : 100,
          "n_transient" : 35855,
          "n_ss" : 8192,
          "num_runs" : 1, 
          }
v = Vodscillator(**params)
print(v)

    





