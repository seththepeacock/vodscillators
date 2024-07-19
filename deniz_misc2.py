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


ref_type = "next_win"
sample_rate=44100
xmin=None
xmax=None
ymin=None
ymax=None
# ymin=0
# ymax=8
show_plot=False
t_win = 0.1
hann = True

ax2 = plt.subplot(2, 1, 1)
coherence_vs_psd(ax=ax2, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=hann)
ax2.set_title("Next window")


ax4 = plt.subplot(2, 1, 2)
ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, ax=ax4, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=hann)
ax4.set_title("Next frequency")

plt.tight_layout()
plt.show()