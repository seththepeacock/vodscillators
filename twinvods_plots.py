from vodscillator import *
from twinvods import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *

filename = "test_twins.pkl"
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\"
with open(filepath + filename, 'rb') as picklefile:
    twins = pickle.load(picklefile)
    assert isinstance(twins, TwinVods)
    sr = twins.sample_rate
    
t_win = 10
fmin = 0
fmax = 10

d = get_mags(wf=twins.P_L, sr=sr, t_win=t_win, dict=True)
mags_P_L = d["mags"]
freqs = d["freq_ax"]
mags_P_R = get_mags(wf=twins.SOO_R, sr=sr, t_win=t_win)

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)


ax1.plot(freqs, mags_P_L)
ax2.plot(freqs, mags_P_R)
for ax in [ax1, ax2]:
    ax.set_xlim(fmin, fmax)
plt.show()



    