from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
from plots import *
from vlodder import *
import scipy.io

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "DejaVuSans"
#     "font."
# })

# get waveforms
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
filename = 'TH14RearwaveformSOAE.mat'
mat = scipy.io.loadmat(filepath + filename)
wf1 = np.squeeze(mat['wf'])
sr1=44100
wf_title1 = "Human SOAE Waveform"

filename = "V&D fig 3A, loc=0.1, glob=0, sr=128.pkl"
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    wf2 = vod.SOO_sol[vod.n_transient:]
sr2=128
wf_title2 = r"V&D Model Simulated Waveform"

# filename = "wf - V&D fig 2A, loc=0.1, glob=0, sr=128.pkl"
# filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
# with open(filepath + filename, 'rb') as picklefile:
#     wf2 = pickle.load(picklefile)
# sr2=128
# wf_title2 = "V\&D Model Waveform (Figure 4 Parameters)"




# set params for TH14
t_win1=0.1
hann1=False
# set params for V&D
t_win2=20
hann2=True

# get coherence, mags, and means
w1 = get_wfft(wf=wf1, sr=sr1, t_win=t_win1, hann=hann1)
wfft1 = w1["wfft"]
freq_ax1 = w1["freq_ax"]

w2 = get_wfft(wf=wf2, sr=sr2, t_win=t_win2, hann=hann2)
wfft2 = w2["wfft"]
freq_ax2 = w2["freq_ax"]

# we'll pass the wfft and its freq axis into these so we don't have to get them again
c1 = get_coherence(wfft=wfft1, freq_ax=freq_ax1, wf=wf1, sr=sr1, ref_type="next_freq", t_win=t_win1, return_all=True)
m1 = get_mags(wfft=wfft1, freq_ax=freq_ax1, wf=wf1, sr=sr1, t_win=t_win1, return_all=True)

c2 = get_coherence(wfft=wfft2, freq_ax=freq_ax2, wf=wf2, sr=sr2, ref_type="next_freq", t_win=t_win2, return_all=True)
m2 = get_mags(wfft=wfft2, freq_ax=freq_ax2, wf=wf2, sr=sr2, t_win=t_win2, return_all=True)

# set up axes
fig1 = plt.figure(1)
axt1 = plt.subplot(2, 1, 1)
axb1 = plt.subplot(2, 1, 2)
fig2 = plt.figure(2)
axt2 = plt.subplot(2, 1, 1)
axb2 = plt.subplot(2, 1, 2)

# define a helper function which we'll call for each waveform. c and p are the coherence/mags dictionaries
def plot(c, m, axt, axb, khz, wf):
    # get vars from the dictionaries
    mags = m["mags"]
    mags_freq_ax = m["freq_ax"]
    coherence = c["coherence"]
    coherence_freq_ax = c["freq_ax"]
    means = c["means"]
    
    # normalize by pi
    # means = means/np.pi
    
    # convert to db/khz
    mags = 20*np.log10(mags)
    if khz:
        mags_freq_ax = mags_freq_ax / 1000
        coherence_freq_ax = coherence_freq_ax / 1000
    
    # set marker for PC/Magnitude
    markA = "+"
    # and for means
    markB = "."
    # set default font size
    fs = "18"
    # define the <|phase diffs|> string
    means_label = r"$\langle|\phi_j^{{\theta}}|\rangle$"
    # twin the axes so we can add a second y axis to each subplot
    axt2 = axt.twinx()
    axb2 = axb.twinx()
    
    # TOP SUBPLOT
    # plot PC
    axt2.plot(coherence_freq_ax, coherence, label=r"$C_{{\theta}}$", marker=markA, color='m', lw=1)
    axt2.set_ylabel('Vector Strength', fontsize=fs)
    axt2.legend(loc="upper right", fontsize=fs)
    if wf=="V+D":
        axt2.legend(loc="lower right", fontsize=fs)
    
    # plot means
    axt.plot(coherence_freq_ax, means, label=means_label, color='black', marker=markB, lw=1)
    axt.set_ylabel(means_label, fontsize=fs)
    axt.legend(loc="upper left", fontsize=fs)
    if wf=="V+D":
        axt.legend(loc="lower left", fontsize=fs)

    # BOTTOM SUBPLOT
    # plot mags 
    axb2.plot(mags_freq_ax, mags, label="Magnitude", color='r', marker=markA, lw=1)
    axb2.set_ylabel('Magnitude [dB]', fontsize=fs)
    axb2.legend(loc="lower right", fontsize=fs)
    
    # plot means
    axb.plot(coherence_freq_ax, means, label=means_label, color='black', marker=markB, lw=1)
    axb.set_ylabel(means_label, fontsize=fs)
    axb.legend(loc="lower left", fontsize=fs)
    
    # set xlabels
    axt.set_xlabel("Frequency [kHz]", fontsize=fs)
    axb.set_xlabel("Frequency [kHz]", fontsize=fs)



# call the function on both waveforms
plot(c1, m1, axt1, axb1, True, "TH14")
plot(c2, m2, axt2, axb2, False, "V+D")

# set titles
# title = r"$\langle|\phi_j^{{\theta}}|\rangle$" + ", " + r"$C_{{\theta}}$" + ", and Magnitude for "
title = ""
axt1.set_title(title + wf_title1, fontsize="22")
axt2.set_title(title + wf_title2, fontsize="22")

# set lims
fmax1 = 5
fmax2 = 7
axt1.set_xlim(0, fmax1)
axb1.set_xlim(0, fmax1)
axt2.set_xlim(0, fmax2)
axb2.set_xlim(0, fmax2)
axt1.set_ylim(0, np.pi)
axb1.set_ylim(0, np.pi)
axt2.set_ylim(0, np.pi)
axb2.set_ylim(0, np.pi)

# finalize
plt.tight_layout()
fig1.set_size_inches(18, 10) # set figure's size manually
fig2.set_size_inches(18, 10)
fig1.savefig('abs_avg_pd_TH14.png', dpi=500, bbox_inches='tight')
fig2.savefig('abs_avg_pd_V+D.png', dpi=500, bbox_inches='tight')
plt.show()