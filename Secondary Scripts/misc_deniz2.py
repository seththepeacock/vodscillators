from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io
from scipy.signal import *

#filename = "V&D fig 2A, loc=0, glob=0.pkl"
#filename = "V&D fig 5, loc=0.0785, glob=0, sr=128.pkl"
#filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
#with open(filepath + filename, 'rb') as picklefile:
#    vod = pickle.load(picklefile)
    #this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
#    assert isinstance(vod, Vodscillator)
#
#vod.n_win = vod.n_ss
#vod.save()

#filename = 'AC6rearSOAEwfB1.mat'
filename = 'TH14RearwaveformSOAE.mat'
mat = scipy.io.loadmat('SOAE Data/' + filename)
wf = np.squeeze(mat['wf'])
wf_title = filename

#wf = vod.SOO_sol[vod.n_transient:]

#wf = np.pad(wf, 20) #padding might help???


#sample_rate=128
sample_rate = 44100 #for SOAE data
win_size = 2048
xmin=0
xmax=None
ymin=None
ymax=None
# ymin=0
# ymax=8
#t_win = win_size / sample_rate
t_win = 0.1
hann = True
t_shift = t_win  #set this to half the window size #it used to be 0.1
khz=True

fig, _ = plt.subplots(2, 2)
axes = fig.axes


#fig.rcParams.update({
#    "text.usetex": True,
#    "font.family": "DejaVuSans"
#    "font."
#})


fig.suptitle(str((filename) + ", sample_rate=" + str(sample_rate) + ", win_size=" + str(win_size)))




ref_type = "next_win"
coherence_vs_psd(ax=axes[0], t_shift=t_shift, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sr=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, hann=False)
axes[0].legend()
axes[0].set_title(r"$C_{\tau}$ (without Hann window)")


ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, ax=axes[1], wf=wf, ref_type=ref_type, t_win=t_win, sr=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, hann=False)
axes[1].legend()
axes[1].set_title(r"$C_{\theta}$ (without Hann window)")


ref_type = "next_win"
coherence_vs_psd(ax=axes[2], t_shift=t_shift, wf_title=wf_title, wf=wf, ref_type=ref_type, t_win=t_win, sr=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, hann=hann)
axes[2].legend()
axes[2].set_title(r"$C_{\tau}$ (with Hann window)")


ref_type="next_freq"
coherence_vs_psd(wf_title=wf_title, t_shift=t_shift, ax=axes[3], wf=wf, ref_type=ref_type, t_win=t_win, sr=sample_rate,
                 xmin=xmin, xmax=xmax, khz=khz, hann=hann)
axes[3].legend()
axes[3].set_title(r"$C_{\theta}$ (without Hann window)")

dpi=300
reso=[16, 9]
bbox="tight"
plt.gcf().set_size_inches(reso) # set figure's size manually to your full screen (32x18)

save_name = str("Hann comparison " + (filename) + ", sample_rate=" + str(sample_rate) + ", win_size=" + str(win_size) + ".png")
plt.savefig(save_name, dpi=dpi, bbox_inches=bbox)
plt.show()


#print(save_name)