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
ref_type="next_win"
t_win = 24
t_shift = 0.025
scope = 1
ref_type="next_win"
coherogram(wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, scope=scope, t_shift=t_shift, sample_rate=sample_rate, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=2)
plt.show()
t_win = 1
t_shift = 0.025
coherence_vs_psd(wf_title=wf_title, wf=wf, t_win=t_win, t_shift=t_shift, sample_rate=sample_rate, xmin=100, xmax=20000, show_plot=show_plot, fig_num=3)
plt.show()

# t_win = 24
# t_shift = 0.025
# # vmin=-40
# vmin=None
# spectrogram(wf_title=wf_title, wf=wf, t_win=t_win, db=True, t_shift=t_shift, vmin=vmin, sample_rate=sample_rate, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=4)
# plt.show()

# t_win = 24
# t_shift = 4
# scope = 
# ref_type="next_freq"
# freq_ref_step=1
# coherogram(wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, freq_ref_step=freq_ref_step, scope=scope, t_shift=t_shift, sample_rate=sample_rate, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=1)
# plt.show()


