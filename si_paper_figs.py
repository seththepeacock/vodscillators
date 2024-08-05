from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
from plots import *
from vlodder import *
import scipy.io


# get waveforms
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
filename = 'TH14RearwaveformSOAE.mat'
mat = scipy.io.loadmat(filepath + filename)
wf1 = np.squeeze(mat['wf'])
sr1=44100
wf_title1 = "Human SOAE Waveform"

filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
with open(filepath + filename, 'rb') as picklefile:
    wf2 = pickle.load(picklefile)
    wf2=wf2[0:int(len(wf2)/8)]
sr2=128
wf_title2 = "V&D Simulated Waveform"

# set up axes
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(2, 2, 2)
ax4 = plt.subplot(2, 2, 4)

# set params for TH14
xmin=0
xmax=5000
ymin=None
ymax=None
t_win=0.1
bin_shift=1
hann=False

do_means=True
do_coherence=True
do_psd=False
coherence_vs_psd(ax=ax1, wf=wf1, wf_title=wf_title1, sample_rate=sr1, ref_type="next_freq", bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, do_coherence=do_coherence, do_psd=do_psd, do_means=do_means, hann=hann)
do_coherence=False
do_psd=True
coherence_vs_psd(ax=ax2, wf=wf1, wf_title=wf_title1, sample_rate=sr1, ref_type="next_freq", bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, do_coherence=do_coherence, do_psd=do_psd, do_means=do_means, hann=hann)

# set params for V&D
xmin=0
xmax=7
ymin=None
ymax=None
t_win=15
bin_shift=1
hann=True

do_means=True
do_coherence=True
do_psd=False
coherence_vs_psd(ax=ax3, wf=wf2, wf_title=wf_title2, sample_rate=sr2, ref_type="next_freq", bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, do_coherence=do_coherence, do_psd=do_psd, do_means=do_means, hann=hann)
do_coherence=False
do_psd=True
coherence_vs_psd(ax=ax4, wf=wf2, wf_title=wf_title2, sample_rate=sr2, ref_type="next_freq", bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, do_coherence=do_coherence, do_psd=do_psd, do_means=do_means, hann=hann)



plt.show()