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
filename = "FB_sr_512_noniso.pkl"


with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)


pp.pprint(vod.params())