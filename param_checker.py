import numpy as np
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from itertools import combinations
from vodscillator import *
import pprint as pp


filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
filename = "F&B fig 2D, iso, 'beta_sigma': 0.0, 'glob_noise_amp': 0.1, 'loc_noise_amp': 0.1, 'num_wins': 30, 'num_osc': 50, 'sample_rate': 512.pkl"


with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)

params = vod.params()

#filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
#filename = "F&B fig 2D noniso.pkl"
#
#with open(filepath + filename, 'rb') as picklefile:
#    vod2 = pickle.load(picklefile)
#    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
#    assert isinstance(vod, Vodscillator)
#
#

#toprint = [(i, vod.params()[i], vod2.params()[i]) for i in vod.params().keys() if vod.params()[i] != vod2.params()[i]]

#pp.pprint(vod.params())
pp.pprint(vod.params(['num_wins', 'sample_rate', 'num_osc', 'glob_noise_amp', 'loc_noise_amp', 'beta_sigma']))