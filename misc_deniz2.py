from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io
from scipy.signal import *

#filename = "V&D fig 2A, loc=0, glob=0.pkl"
#filename = "V&D fig 2A, loc=0.1, glob=0.1.pkl"
filename = "V&D fig 5, loc=0.0785, glob=0, sr=128.pkl"
filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    #this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)
#
#vod.n_win = vod.n_ss
#vod.save()

#filename = 'AC6rearSOAEwfB1.mat'
#filename = 'TH14RearwaveformSOAE.mat'
#mat = scipy.io.loadmat('SOAE Data/' + filename)
#wf = np.squeeze(mat['wf'])
wf_title = filename

wf = vod.SOO_sol

#wf = np.pad(wf, 20) #padding might help???


sample_rate=44100
win_size = 2048
xmin=0
xmax=7
ymin=None
ymax=None
# ymin=0
# ymax=8
show_plot=False
#t_win = win_size / sample_rate
t_win = 0.05
hann = True
t_shift = t_win / 2 #set this to half the window size #it used to be 0.1
fcut=False
khz=True

fig, _ = plt.subplots(2, 2)
axes = fig.axes

fig.suptitle(str((filename) + ", sample_rate=" + str(sample_rate) + ", win_size=" + str(win_size)))




ref_type = "next_win"
coherence_vs_psd(ax=axes[0], t_shift=t_shift, fcut=fcut, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, show_plot=show_plot, hann=False)
axes[0].legend(loc='upper center')
axes[0].set_title("Next window (no Hann)")


ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, fcut=fcut, ax=axes[1], wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, show_plot=show_plot, hann=False)
axes[1].legend(loc='upper center')
axes[1].set_title("Next frequency (no Hann)")


ref_type = "next_win"
coherence_vs_psd(ax=axes[2], t_shift=t_shift, fcut=fcut, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, show_plot=show_plot, hann=hann)
axes[2].legend(loc='upper center')
axes[2].set_title("Next window (with Hann)")


ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, fcut=fcut, ax=axes[3], wf=wf, ref_type=ref_type, t_win=t_win, sample_rate=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, show_plot=show_plot, hann=hann)
axes[3].legend(loc='upper center')
axes[3].set_title("Next frequency (with Hann)")

dpi=300
reso=[16, 9]
bbox="tight"
plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)

save_name = str((filename) + ", sample_rate=" + str(sample_rate) + ", win_size=" + str(win_size) + ".png")
plt.savefig(save_name, dpi=dpi, bbox_inches=bbox)
plt.show()


#print(save_name)