from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io
from scipy.signal import welch
#from scipy.signal import *

#filename = "V&D fig 2A, loc=0.1, glob=0.1.pkl"
#filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"


#with open(filepath + filename, 'rb') as picklefile:
#    vod = pickle.load(picklefile)
#    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
#    assert isinstance(vod, Vodscillator)

#vod.n_win = vod.n_ss

# vod.save()


filename = 'TH14RearwaveformSOAE'
mat = scipy.io.loadmat('SOAE Data/' + 'TH14RearwaveformSOAE.mat')
wf = np.squeeze(mat['wf'])

sample_rate=44100
t_win = 24

wfft2 = get_wfft2(wf, sample_rate, t_win, return_all = False)
coh = get_coherence(wf, sample_rate, t_win, return_all=True, wfft=wfft2)
coherence = coh['coherence']
freq_ax = coh['freq_ax']
plt.plot(freq_ax, coherence)
plt.show()