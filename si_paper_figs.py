from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
from plots import *
from vlodder import *
import scipy.io

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVuSans"
    "font."
})

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
wf_title2 = "V\&D Model Waveform (Figure 5 Parameters)"


# set params for TH14
t_win1=0.1
hann1=False
# set params for V&D
t_win2=10
hann2=True

# get coherence, psd, and means
w1 = get_wfft(wf=wf1, sr=sr1, t_win=t_win1, hann=hann1)
wfft1 = w1["wfft"]
freq_ax1 = w1["freq_ax"]

w2 = get_wfft(wf=wf2, sr=sr2, t_win=t_win2, hann=hann2)
wfft2 = w2["wfft"]
freq_ax2 = w2["freq_ax"]

# we'll pass the wfft and its freq axis into these so we don't have to get them again
c1 = get_coherence(wfft=wfft1, freq_ax=freq_ax1, wf=wf1, sr=sr1, ref_type="next_freq", t_win=t_win1, return_all=True)
p1 = get_psd(wfft=wfft1, freq_ax=freq_ax1, wf=wf1, sr=sr1, t_win=t_win1, return_all=True)

c2 = get_coherence(wfft=wfft2, freq_ax=freq_ax2, wf=wf2, sr=sr2, ref_type="next_freq", t_win=t_win2, return_all=True)
p2 = get_psd(wfft=wfft2, freq_ax=freq_ax2, wf=wf2, sr=sr2, t_win=t_win2, return_all=True)

# set up axes
fig1 = plt.figure(1)
axt1 = plt.subplot(2, 1, 1)
axb1 = plt.subplot(2, 1, 2)
# fig2 = plt.figure(2)
# axt2 = plt.subplot(2, 1, 1)
# axb2 = plt.subplot(2, 1, 2)

# define a helper function which we'll call for each waveform. c and p are the coherence/psd dictionaries
def plot(c, p, axt, axb, khz):
    # get vars from the dictionaries
    psd = p["psd"]
    psd_freq_ax = p["freq_ax"]
    coherence = c["coherence"]
    coherence_freq_ax = c["freq_ax"]
    means = c["means"]
    
    # normalize by pi
    # means = means/np.pi
    
    # convert to db/khz
    psd = 20*np.log10(psd)
    if khz:
        psd_freq_ax = psd_freq_ax / 1000
        coherence_freq_ax = coherence_freq_ax / 1000
    
    # set marker for PC/PSD
    markA = "+"
    # and for means
    markB = "."
    # set legend font size
    fs = "18"
    # define the <|phase diffs|> string
    means_label = r"$\langle|\phi_j^{{\theta}}|\rangle$"
    # twin the axes so we can add a second y axis to each subplot
    axt2 = axt.twinx()
    axb2 = axb.twinx()
    
    #TOP SUBPLOT
    # plot PC
    axt2.plot(coherence_freq_ax, coherence, label=r"$C_{{\theta}}$", marker=markA, color='m', lw=1)
    axt2.set_ylabel('Vector Strength', fontsize=fs)
    axt2.legend(loc="upper right", fontsize=fs)
    
    # plot means
    axt.plot(coherence_freq_ax, means, label=means_label, color='black', marker=markB, lw=1)
    axt.set_ylabel(means_label, fontsize=fs)
    axt.legend(loc="upper left", fontsize=fs)

    #BOTTOM SUBPLOT
    # plot psd 
    axb2.plot(psd_freq_ax, psd, label="PSD", color='r', marker=markA, lw=1)
    axb2.set_ylabel('PSD [dB]', fontsize=fs)
    axb2.legend(loc="lower right", fontsize=fs)
    
    # plot means
    axb.plot(coherence_freq_ax, means, label=means_label, color='black', marker=markB, lw=1)
    axb.set_ylabel(means_label, fontsize=fs)
    axb.legend(loc="lower left", fontsize=fs)
    
    # set xlabels
    axt.set_xlabel("Frequency [kHz]", fontsize=fs)
    axb.set_xlabel("Frequency [kHz]", fontsize=fs)



# call the function on both waveforms
plot(c1, p1, axt1, axb1, True)
# plot(c2, p2, axt2, axb2, False)

# set titles
title = r"$\langle|\phi_j^{{\theta}}|\rangle$" + ", " + r"$C_{{\theta}}$" + ", and PSD for "
axt1.set_title(title + wf_title1, fontsize="22")
# axt2.set_title(title + wf_title2, fontsize="22")

# set lims
fmax1 = 5
fmax2 = 7
axt1.set_xlim(0, fmax1)
axb1.set_xlim(0, fmax1)
# axt2.set_xlim(0, fmax2)
# axb2.set_xlim(0, fmax2)

# finalize
plt.tight_layout()
fig1.set_size_inches(18, 10) # set figure's size manually
# fig2.set_size_inches(18, 10)
plt.show()
# fig1.savefig('abs_avg_pd_TH14.png', dpi=500, bbox_inches='tight')
# fig2.savefig('abs_avg_pd_V+D.png', dpi=500, bbox_inches='tight')