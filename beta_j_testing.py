from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from plots import *
import scipy.io
from vlodder import *

filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
filename = "FB_sr_512_noniso.pkl"


with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)

params = vars(vod)

beta_j_list = params['betas']


oscillators = np.arange(0, vod.num_osc, 1)

plt.figure()
plt.plot(oscillators, beta_j_list)
plt.xlabel("Oscillator index")
plt.ylabel("beta_j")
plt.title("deniz " + filename)
plt.show()