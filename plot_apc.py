from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from plots import *
from vlodder import *

#/home/deniz/Dropbox/vodscillators/cluster_width=0.1, num_wins=100, t_win=0.5, amp_weights=False.pkl

cluster_width=0.1
f_resolution=0.01
num_wins=100
t_win=1
amp_weights= True
f_min=0
f_max=7

cc_t_win = 24
cc_t_shift = 1

#open our stuff
filename = f"cluster_width={cluster_width}, num_wins={num_wins}, t_win={t_win}, amp_weights={amp_weights}.pkl"
# filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\APC V&D fig 2A, loc=0, glob=0\\"
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\APC V&D fig 4, loc=0.1, glob=0, sr=128\\"
# load apc data
with open(filepath + filename, 'rb') as picklefile:
    apc = pickle.load(picklefile)
# get freq ax for the apc phase coherence
apc_freq_ax = np.arange(f_min, f_max, f_resolution)

# load vodscillator for PSD and classic phase coherence
# vod_file= "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\V&D fig 2A, loc=0, glob=0.pkl"
# vod_file= "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\V&D fig 2A, loc=0.1, glob=0.pkl"
# vod_file= "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\F&B fig 2D, iso, loc=0.1, glob=0.pkl"
vod_file= "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\V&D fig 4, loc=0.1, glob=0, sr=128.pkl"

with open(vod_file, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)
    # any new vod pickles will have this predefined, but you'll have to calculate it by hand for now!
    # vod.t_transient = vod.n_transient / vod.sample_rate
    # vod.n_win = vod.n_ss
    # vod.name="F&B fig 2D, iso, loc=0.1, glob=0.pkl"
    # vod.save()

# get 2 axes for double y axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# get the classic phase coherence with variable t_win
c = get_coherence(vod.SOO_sol[vod.n_transient:], vod.sample_rate, t_win=cc_t_win, t_shift=cc_t_shift, return_all=True)
classic_coherence = c["coherence"]
cc_freq_ax = c["freq_ax"]

# we'll get the psd from the vod method since the fft is precalculated, and we don't need to mess aroudn with the window size
psd = get_psd_vod(vod)

# plot
ax1.plot(apc_freq_ax, apc, label=f"APC: cluster_width={cluster_width}, num_wins={num_wins}, t_win={t_win}, amp_weights={amp_weights}", color='b')
ax1.plot(cc_freq_ax, classic_coherence, label="Classic Coherence", color='purple')
ax2.plot(vod.fft_freq, 10*np.log10(psd), label="PSD", color='r')

# set labels
ax1.set_ylabel('Phase Coherence', color='b')
ax2.set_xlabel('Freq')
ax2.set_ylabel('PSD [dB]', color='r')

# set title, show legend, set xlims
plt.title("Comparison of PSD and PC for F&B fig 2D, noniso, loc=0.1, glob=0")
ax1.legend(loc="lower left")
ax2.legend(loc="lower right")
# ax2.set_ylim(-15, 60)
plt.xlim(f_min, f_max)
plt.show()