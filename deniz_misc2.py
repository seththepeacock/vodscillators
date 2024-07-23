from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io
from scipy.signal import *

#filename = "F&B fig 2D iso.pkl"
##filename = "F&B fig 2D noniso.pkl"
#filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
#with open(filepath + filename, 'rb') as picklefile:
#    vod = pickle.load(picklefile)
#    #this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
#    assert isinstance(vod, Vodscillator)
#
##vod.n_win = vod.n_ss
##vod.save()

#filename = 'AC6rearSOAEwfB1.mat'
filename = 'TH14RearwaveformSOAE.mat'
mat = scipy.io.loadmat('SOAE Data/' + filename)
wf = np.squeeze(mat['wf'])
wf_title = filename

#wf = vod.SOO_sol


sample_rate=44100
win_size = 4096
xmin=0
xmax=5
ymin=None
ymax=None
# ymin=0
# ymax=8
show_plot=False
t_win = win_size / sample_rate
hann = True
t_shift = t_win / 2 #set this to half the window size #it used to be 0.1


fig, (ax1, ax2) = plt.subplots(2) #no hann
figg, (ax3, ax4) = plt.subplots(2) #yes hann
fig.suptitle('No Hann')
figg.suptitle("Yes Hann")

ref_type = "next_win"
coherence_vs_psd(ax=ax1, t_shift=t_shift, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=False)
ax1.legend(loc='upper right')
ax1.set_title("Next window ")


ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, ax=ax2, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=False)
ax2.legend(loc='upper right')
ax2.set_title("Next frequency ")


ref_type = "next_win"
coherence_vs_psd(ax=ax3, t_shift=t_shift, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=hann)
ax3.legend(loc='upper right')
ax3.set_title("Next window")


ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, ax=ax4, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=True, show_plot=show_plot, hann=hann)
ax4.legend(loc='upper right')
ax4.set_title("Next frequency")


plt.show()
