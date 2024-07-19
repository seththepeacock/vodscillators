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


filename = 'AC6rearSOAEwfB1.mat'
mat = scipy.io.loadmat('SOAE Data/' + 'AC6rearSOAEwfB1.mat')
wf = np.squeeze(mat['wf'])
wf_title = filename
    
# global
sample_rate=44100
xmin=None
xmax=None
ymin=None
ymax=None
# ymin=0
# ymax=8
show_plot=False
t_win = 2
coherence_vs_psd2(wf_title=wf_title, wf=wf, t_win=t_win, sample_rate=sample_rate, xmin=0, xmax=10, show_plot=show_plot, fig_num=3)
plt.show()
