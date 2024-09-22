from vodscillator import *
from twinvods import *
from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io



filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
filename = "twins.pkl"


with open(filepath + filename, 'rb') as picklefile:
    twins = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Twins couple, so it will display its documentation for you!
    assert isinstance(twins, Twins)


twins.save()


# ALL VALUES <|phase diff|> vs coherence (V&D)
if 1==0: 
    # filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # with open(filepath + filename, 'rb') as picklefile:
    #     wf = pickle.load(picklefile)
    #     wf = wf[0:int(len(wf))]
    # sample_rate=128
    # wf_title = "V&D fig 4 (loc=0.1, glob=0)"
    # xmin=0
    # xmax=6.5
    # t_win=10

    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
    # filename = 'TH14RearwaveformSOAE.mat'
    filename = 'ACsb24rearSOAEwfA1'
    mat = scipy.io.loadmat(filepath + filename)
    wf = np.squeeze(mat['wf'])
    sample_rate=44100
    xmin=1000
    xmax=5000
    
    ymin=None
    ymax=None
    wf_title=filename
    t_win=0.02
    bin_shift=1
    
    # set global
    do_means=False
    

    # plt.figure(1)
    # ax1 = plt.gca()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    quick(ax=ax1, wf=wf, wf_title=wf_title, sample_rate=sample_rate, ref_type="next_freq", do_means=do_means, bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # bin_shift=3
    # t_win=0.02
    do_means=True
    quick(do_coherence=False, ax=ax2, wf=wf, wf_title=wf_title, sample_rate=sample_rate, ref_type="next_freq", do_means=do_means, bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) 
    plt.tight_layout()
    plt.show()
    


    # c = get_coherence(wf, ref_type="next_freq", sample_rate=sample_rate, t_win=t_win, bin_shift=bin_shift, return_all=True)
    # num_wins = c["num_wins"]
    # phase_diffs = c["phase_diffs"]
    # freq_ax_n = c["freq_ax"]
    
    # # get the mean over all windows (0th axis)
    # means_n = np.mean(np.abs(phase_diffs[:, :]), 0)
    # ax2.plot(freq_ax_n, means_n, label="<|Phase Diff (Next)|>")
    # ax2.set_xlim(xmin, xmax)
    # ax2.set_title(f"Average of Absolute Value of Phase Diffs (Next Freq) for {wf_title}")
    # ax2.set_xlabel("Frequency")
    # ax2.set_ylabel("<|Phase Diff|>")
    # ax2.legend()
    # plt.tight_layout()


# plotting twins
if 1==1:
    filename = "twins.pkl"
    filepath = "/home/deniz/Dropbox/vodscillators/deniz pickle jar/"
    with open(filepath + filename, 'rb') as picklefile:
        twins = pickle.load(picklefile)
        assert isinstance(twins, Twins)

    fig, _ = plt.subplots(2, 2)
    axes = fig.axes
    
    wfs = [twins.SOOL_sol, twins.SOOR_sol, twins.T_l_sol, twins.T_r_sol]
    wf_titles = ["SOOL", "SOOR", "T_l", "T_r"]
    t_win = 10
    t_shift = t_win
    
    for i in range(4):
        wf = wfs[i][twins.n_transient:]
        quick(wfs[i], sample_rate=twins.sample_rate, t_win=t_win, t_shift=t_shift, ax=axes[i], wf_title=wf_titles[i])
        axes[i].set_xlim(0, 8)
        
    plt.show()
    
    



# fft plotting
if 1 == 0:
    fftR = twins.right_SOO_fft
    fftL = twins.left_SOO_fft
    freqpoints = twins.fft_freq
    plt.plot(freqpoints,fftR[0])
    plt.show()


