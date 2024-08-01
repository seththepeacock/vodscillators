from vodscillator import *
from twins_mech import *
from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io



filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
filename = "twins.pkl"


with open(filepath + filename, 'rb') as picklefile:
    twins = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Twins couple, so it will display its documentation for you!
    assert isinstance(twins, Twins)

twins.do_fft()
twins.save()

fftR = twins.right_SOO_fft
fftL = twins.left_SOO_fft
freqpoints = twins.fft_freq

plt.plot(freqpoints,fftR[0])
plt.show()


