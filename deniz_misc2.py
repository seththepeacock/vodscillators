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
t_shift = 0.1

ax1 = plt.subplot(2, 2, 1)
ref_type = "next_win"
coherence_vs_psd(ax=ax1, t_shift=t_shift, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=False)
ax1.set_title("Next window ")



ax2 = plt.subplot(2, 2, 2)
ref_type = "next_win"
coherence_vs_psd(ax=ax2, t_shift=t_shift, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=hann)
ax2.set_title("Next window Hann")

ax3 = plt.subplot(2, 2, 3)
ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, ax=ax3, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=False)
ax3.set_title("Next frequency ")



ax4 = plt.subplot(2, 2, 4)
ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, ax=ax4, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=hann)
ax4.set_title("Next frequency Hann")

#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
#ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
#ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
#ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
#
#

plt.tight_layout()
plt.show()