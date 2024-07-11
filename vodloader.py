import numpy as np
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from itertools import combinations
from vodscillator import *


filepath = "/home/deniz/Dropbox/vodscillators/"
filename = "FB_sr_512_noniso.pkl"


with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)



filepath = "/home/deniz/Dropbox/vodscillators/"
filename = "F&B fig 2D noniso.pkl"

with open(filepath + filename, 'rb') as picklefile:
    vod2 = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)



print(vars(vod) == vars(vod2))





#vod.freq_dist = p["freq_dist"] # set to "linear" for frequency increasing linearly, set to "exp" for frequency increasing exponentially
#vod.roughness_amp = p["roughness_amp"] # variation in each oscillator's characteristic frequency
#vod.omega_0 = p["omega_0"]  # char frequency of lowest oscillator [default = 2*np.pi] 
#vod.omega_N = p["omega_N"]  # char frequency of highest oscillator [default = 5*(2*np.pi)] 
#vod.IC_method = p["IC_method"]  # set to "rand" for randomized initial conditions, set to "const" for constant initial conditions
#vod.beta_sigma = p["beta_sigma"] # standard deviation for imaginary coefficient for cubic nonlinearity
#vod.epsilon = p["epsilon"] # [default = 1.0] --> control parameter
#vod.d_R = p["d_R"]  # [default = 0.15] --> real part of coupling coefficient
#vod.d_I = p["d_I"]  # [default = -1.0] --> imaginary part of coupling coefficient
#vod.alpha = p["alpha"] # [default = 1.0] --> real coefficient for cubic nonlinearity
## Tonotopic Frequency Distribution
## Now we set the frequencies of each oscillator in our chain - linear or exponential
#if s.freq_dist == "linear":
#  s.omegas = np.linspace(s.omega_0,s.omega_N,s.num_osc) # linearly spaced frequencies from omega_0 to omega_N
#elif s.freq_dist == "exp":
#  s.omegas = np.zeros(s.num_osc, dtype=float)
#  for k in range(s.num_osc): # exponentially spaced frequencies from omega_0 to omega_N
#    A = s.omega_0
#    B = (s.omega_N/s.omega_0)**(1/(s.num_osc - 1))
#    s.omegas[k] = A*(B**k)
#    # note that omegas[0] = A = omega_0, and omegas[num_osc - 1] = A * B = omega_N 
## add roughness to the freq distribution
#r = s.roughness_amp
#roughness = np.random.uniform(low=-r, high=r, size=(s.num_osc,))
#s.omegas = s.omegas + roughness
## Generate initial conditions and store in ICs[]
#s.ICs = np.zeros(s.num_osc, dtype=complex)