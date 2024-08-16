from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
from twins_mech import *
import scipy.io
import random as rand
# vod.n_win = vod.n_ss
# vod.save()

# testing human reversal with t_shift
if 1==1:
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
    filename = 'TH14RearwaveformSOAE.mat'
    # filename = 'ACsb24rearSOAEwfA1'
    mat = scipy.io.loadmat(filepath + filename)
    wf = np.squeeze(mat['wf'])
    wf_title=filename
    sr=44100
    # xmin=2.17
    # xmax=2.19
    xmin=1
    xmax=5
    
    t_win = 0.046
    t_shift = t_win
    
    fig, _ = plt.subplots(3, 2)
    axes = fig.get_axes()
    
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[0], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_win = 0.020
    t_shift = 0.020
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[1], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_win = 0.02
    t_shift = 0.01
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[2], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_shift = 0.005
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[3], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_shift = 0.003
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[4], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_shift = 0.001
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[5], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    
    
    
    for ax in axes:
        ax.set_title("")
    fig.suptitle("Human", fontsize="20")
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    fig.savefig('C_tau with t_shift - Human', dpi=500, bbox_inches='tight')
    plt.show()


# testing anolis reversal with t_shift
if 1==0:
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
    # filename = 'TH14RearwaveformSOAE.mat'
    filename = 'ACsb24rearSOAEwfA1'
    mat = scipy.io.loadmat(filepath + filename)
    wf = np.squeeze(mat['wf'])
    wf_title=filename
    sr=44100
    # xmin=2.17
    # xmax=2.19
    xmin=1
    xmax=5
    
    t_win = 0.023
    t_shift = t_win
    
    fig, _ = plt.subplots(3, 2)
    axes = fig.get_axes()
    
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[0], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_win = 0.0058
    t_shift = 0.0058
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[1], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_win = 0.01
    t_shift = 0.0058
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[2], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_shift = 0.003
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[3], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_shift = 0.001
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[4], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    t_shift = 0.0005
    coherence_vs_psd(wf, sr, t_win, t_shift=t_shift, ax=axes[5], khz=True, xmin=xmin, xmax=xmax, wf_title=wf_title)
    
    
    
    for ax in axes:
        ax.set_title("")
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    fig.suptitle("Anolis", fontsize="20")
    fig.savefig('C_tau with t_shift - Anolis', dpi=500, bbox_inches='tight')
    plt.show()


# testing next_freq vs prev_freq vs both_freqs
if 1==0:
    filename = 'TH14RearwaveformSOAE'
    mat = scipy.io.loadmat('SOAE Data/' + 'TH14RearwaveformSOAE.mat')
    wf = np.squeeze(mat['wf'])
    wf = wf[0:int(len(wf)/8)]
    wf_title = filename
    
    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)
    # coherence_vs_psd(ax=ax1, wf=wf, sr=44100, t_win=0.1, ref_type="next_freq", xmin=4, xmax=6, khz=True)
    # coherence_vs_psd(ax=ax2, wf=wf, sr=44100, t_win=0.1, ref_type="both_freqs", xmin=4, xmax=6, khz=True)
    # plt.show()
    scatter_phase_diffs(freq=2900, wf=wf, sr=44100, t_win=0.1, wf_title="TH14 WRAPPED")
    plt.show()
    
# psd + coherence of vodscillators
if 1==0:
    # Open pickled vodscillator
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # filename = "V&D fig 5, loc=0.0785, glob=0, sr=128.pkl"
    # with open(filepath + filename, 'rb') as picklefile:
    #     vod = pickle.load(picklefile)
    #     # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    #     assert isinstance(vod, Vodscillator)
    # wf_title=filename
    # wf = np.sum(vod.sol[:, vod.n_transient:], 0)
    
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)
        wf=wf[:int(len(wf)/8)]
    wf_title = "V&D fig 4, loc=0.1, glob=0, sr=128"
    sr = 128
    t_win = 10
    xmin = 0
    xmax = 10
    noise_amp = 1
    wf = wf.real
    mags_normed = rfft(wf, n=1000, norm="forward")
    mags = rfft(wf, n=1000, norm="backward")
    freq = rfftfreq(1000, 1/sr)
    plt.plot(freq, mags/1000 + 1, label="unnormed")

    plt.plot(freq, mags_normed, label="normed")
    plt.xlim(0, 10)
    plt.show()
    

# ALL VALUES <|phase diff|> vs coherence (V&D)
if 1==0: 
    # filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # with open(filepath + filename, 'rb') as picklefile:
    #     wf = pickle.load(picklefile)
    #     wf = wf[0:int(len(wf))]
    # sr=128
    # wf_title = "V&D fig 4 (loc=0.1, glob=0)"
    # xmin=0
    # xmax=6.5
    # t_win=10

    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
    # filename = 'TH14RearwaveformSOAE.mat'
    filename = 'ACsb24rearSOAEwfA1'
    mat = scipy.io.loadmat(filepath + filename)
    wf = np.squeeze(mat['wf'])
    sr=44100
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

    quick(ax=ax1, wf=wf, wf_title=wf_title, sr=sr, ref_type="next_freq", do_means=do_means, bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # bin_shift=3
    # t_win=0.02
    do_means=True
    quick(do_coherence=False, ax=ax2, wf=wf, wf_title=wf_title, sr=sr, ref_type="next_freq", do_means=do_means, bin_shift=bin_shift, t_win=t_win, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) 
    plt.tight_layout()
    plt.show()
    


    # c = get_coherence(wf, ref_type="next_freq", sr=sr, t_win=t_win, bin_shift=bin_shift, return_all=True)
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
if 1==0:
    filename = "test_twins.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
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
        quick(wfs[i], sr=twins.sr, t_win=t_win, t_shift=t_shift, ax=axes[i], wf_title=wf_titles[i])
        axes[i].set_xlim(0, 8)
        
    plt.tight_layout()
    plt.show()
    

# scatter plot for next freq phase diffs
if 1==0: 
    # get wf and set wf sepecific params
    WF = "V&D"
    
    if WF == "TH14":
        filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
        filename = 'TH14RearwaveformSOAE.mat'
        # filename = 'ACsb24rearSOAEwfA1'
        mat = scipy.io.loadmat(filepath + filename)
        wf = np.squeeze(mat['wf'])
        sr=44100
        wf_title = filename
        t_win=0.1
    if WF == "AC":
        filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
        filename = 'ACsb24rearSOAEwfA1'
        mat = scipy.io.loadmat(filepath + filename)
        wf = np.squeeze(mat['wf'])
        sr=44100
        wf_title = filename
    if WF == "V&D":
        filename = "V&D fig 3A, loc=0.1, glob=0, sr=128.pkl"
        filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
        with open(filepath + filename, 'rb') as picklefile:
            vod = pickle.load(picklefile)
            wf = vod.SOO_sol[vod.n_transient:]
        sr=128
        wf_title = "V&D fig 3A (loc=0.1, glob=0)"
        t_win=20
        hann=True


    # set params
    # t_win=0.02
    bin_shift=1
    
    # ax1=plt.subplot(2, 1, 1)
    # ax2=plt.subplot(2, 1, 2)
    ax1= plt.gca()

    freq = 2.226

    scatter_phase_diffs(freq, wf, sr, t_win, hann=hann, ref_type="next_freq", bin_shift=bin_shift, t_shift=None, wf_title=wf_title, ax=ax1)
    # scatter_phase_diffs(4.8, wf, sr, t_win, ref_type="next_freq", bin_shift=bin_shift, t_shift=None, wf_title=wf_title, ax=ax2)
    plt.tight_layout()
    plt.show()


# next freq phase diff comparison for peaks vs valleys (V&D)
if 1==0: 
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)
        wf = wf[0:int(len(wf)/4)]

    # set global
    sr=128
    wf_title = filename
    xmin=0
    xmax=6
    ymin=None
    ymax=None
    ymin=0
    ymax=8
    show_plot=False
    t_win=10

    quick(wf, wf_title=wf_title, hann=True, do_means=True, ref_type="next_freq", sr=sr, t_win=t_win, xmin=xmin, xmax=xmax)
    plt.show()

    # c = get_coherence(wf, ref_type="next_freq", sr=sr, t_win=t_win)
    # num_wins = c["num_wins"]
    # phase_diffs = c["phase_diffs"]
    # freq = 1.10
    # freq_bin_index = int(freq*50)
    # print(np.mean(np.abs(phase_diffs[:, freq_bin_index])))

    # plt.scatter(range(num_wins), phase_diffs[:, freq_bin_index])
    # plt.title(f"Next Freq Bin Phase Diffs for the V&D Fig 4 Coherence/Power Valley at {freq}Hz")
    # plt.xlabel("Window #")
    # plt.ylabel("Phase Diff")
    # plt.show()


# Replicating chris' spectro/coherograms for TH14
if 1==0:
    filename = 'TH14RearwaveformSOAE'
    mat = scipy.io.loadmat('SOAE Data/' + 'TH14RearwaveformSOAE.mat')
    wf = np.squeeze(mat['wf'])
    wf_title = filename
        
    # global
    sr=44100
    xmin=None
    xmax=None
    ymin=None
    ymax=None
    ymin=0
    ymax=6
    show_plot=False

    t_win = 0.045
    t_shift = 0.045
    scope = 1
    ref_type="next_win"
    vmin=0.3
    vmax=1
    cmap="viridis"
    coherogram(wf_title=wf_title, cmap=cmap, wf=wf, t_win=t_win, ref_type=ref_type, khz=True, scope=scope, t_shift=t_shift, vmin=vmin, vmax=vmax, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=2)
    ref_type="next_freq"
    coherogram(wf_title=wf_title, cmap=cmap, wf=wf, t_win=t_win, ref_type=ref_type, khz=True, scope=scope, t_shift=t_shift, vmin=vmin, vmax=vmax, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=3)
    plt.show()
    
    
    # note we are doing psd while chris did amps, so our vmin/vmax is slightly different. The t_win is also approximate.
    # t_win = 0.045
    # t_shift = 0.045
    # vmin=-40
    # vmin=None
    # vmax=None
    # vmin=-100
    # vmax=-60
    # cmap="jet"
    # spectrogram(wf_title=wf_title, wf=wf, t_win=t_win, db=True, khz=True, t_shift=t_shift, vmin=vmin, vmax=vmax, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, cmap=cmap, fig_num=4)
    # plt.show()
    
    # t_win = 0.045
    # t_shift = 0.045
    # ref_type="next_win"
    # coherence_vs_psd(wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=100, xmax=20000, show_plot=show_plot, fig_num=3)
    # plt.show()


# Comparing next win with next freq
if 1==0:
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # filename = "V&D fig 2A, loc=0.1, glob=0, sr=128.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load wf
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)
        # assert(isinstance(vod, Vodscillator))
        # truncate it
        # wf = vod.SOO_sol[vod.n_transient:]
        wf = wf[0:int(len(wf)/4)]
        
    # set global
    sr=128
    xmin=0
    xmax=6
    ymin=None
    ymax=None
    ymin=0
    ymax=8
    show_plot=False
    wf_title = "V&D fig 4, loc=0.1, glob=0"


    # PLOT COHERENCE NEXT_FREQ VS NEXT_WIN
    
    xmin=0
    xmax=6
    ymin=0
    ymax=8
    
    plt.figure(1)
    
    
    downsample_freq=1
    
    ref_type="next_freq"
    ax1 = plt.subplot(2, 1, 1)
    t_win = 20
    t_shift = t_win
    quick(ax=ax1, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax)
    ax1.set_title("Referenced to Next Freq (Window Size = 20s)")
    
    ax3 = plt.subplot(2, 1, 2)
    ref_type="next_freq"
    t_win = 30
    t_shift = t_win
    quick(downsample_freq=downsample_freq, ax=ax3, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax)
    ax3.set_title("Referenced to Next Freq (Window Size = 30s)")
    
    plt.tight_layout()
    
    plt.show()
    
    plt.figure(2)

    ref_type="next_win"
    t_win = 10
    t_shift = t_win
    ax2 = plt.subplot(2, 1, 1)
    quick(ax=ax2, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax)
    ax2.set_title("Referenced to Next Window (Small Window Size)")
    
    ax4 = plt.subplot(2, 1, 2)
    ref_type="next_win"
    t_win = 50
    t_shift = t_win
    quick(downsample_freq=downsample_freq, ax=ax4, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax)
    ax4.set_title("Referenced to Next Window (Large Window Size)")
    
    plt.tight_layout()
    plt.show()
    
    
# testing next_freq on generated waveforms
if 1==0:
    def dxdt(x,t,gamma,w0,A,wd):
        x1, x2 = x
        dx1dt = x2
        dx2dt = -gamma*x2 - (w0**2)*x1 + A*np.sin(wd*t)
        return dx1dt, dx2dt
    
    # get waveform

    # filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # # filename = "V&D fig 2A, loc=0.1, glob=0, sr=128.pkl"
    # # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # # load wf
    # with open(filepath + filename, 'rb') as picklefile:
    #     wf = pickle.load(picklefile)
    #     # assert(isinstance(vod, Vodscillator))
    #     # truncate it
    #     # wf = vod.SOO_sol[vod.n_transient:]
    #     wf = wf[0:int(len(wf)/4)]
    
    
        
    # set global
    wf_title = "V&D fig 4, loc=0.1, glob=0"
    sr=128
    xmin=0
    xmax=6
    ymin=None
    ymax=None
    ymin=0
    ymax=8
    show_plot=False
    do_psd = True

    plt.figure(1)
    
    # plot

    t_win=50
    t_shift=t_win
    ref_type="next_win"
    ax1 = plt.subplot(2, 1, 1)
    quick(ax=ax1, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax, do_psd=do_psd)
    
    ref_type="next_freq"
    ax2 = plt.subplot(2, 1, 2)
    quick(ax=ax2, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax, do_psd=do_psd)
    plt.tight_layout()
    plt.show()
    

# all cohero-figs for Chris' V&D
if 1==0:
    # # good params for next_win coherogram!
    # t_win = 4
    # t_shift = 1
    # scope = 5
    
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)
        # assert(isinstance(vod, Vodscillator))
        # truncate it
        # wf = vod.SOO_sol[vod.n_transient:]
        wf = wf[0:int(len(wf)/4)]
        
    # set global
    sr=128
    xmin=0
    xmax=6
    ymin=None
    ymax=None
    ymin=0
    ymax=8
    show_plot=False
    wf_title = "V&D fig 4, loc=0.1, glob=0"
    
    # # PLOT COHEROGRAM NEXT_WIN
    
    # t_win = 45
    # t_shift = 45
    # scope = 4
    # ref_type="next_win"
    # coherogram(wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, scope=scope, t_shift=t_shift, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=2)
    
    # # PLOT COHEROGRAM NEXT_FREQ
    
    # t_win = 50
    # t_shift = 50
    # scope = 1
    # ref_type="next_freq"
    # freq_ref_step=1
    # coherogram(wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, freq_ref_step=freq_ref_step, scope=scope, t_shift=t_shift, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=1)
    # plt.show()



    # PLOT COHERENCE NEXT_FREQ VS NEXT_WIN
    
    xmin=0
    xmax=50
    ymin=0
    ymax=8
    
    plt.figure(1)
    
    wf = 2*np.random.random_sample(5000*sr) - 1
    do_psd = False
    
    t_win=50
    t_shift=t_win
    ref_type="next_win"
    ax1 = plt.subplot(2, 1, 1)
    quick(ax=ax1, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax, do_psd=do_psd)
    
    ref_type="next_freq"
    ax2 = plt.subplot(2, 1, 2)
    quick(ax=ax2, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax, do_psd=do_psd)
    plt.tight_layout()
    plt.show()
    
    # # Plot Spectrogram and Coherogram in one plot
    # plt.figure(2)
    
    # xmin=None
    # xmax=None
    # vmin=None

    # t_win = 24
    # t_shift = 4
    
    # freq_ref_step=1
    # ax1 = plt.subplot(2, 1, 1)
    # spectrogram(ax=ax1, wf_title=wf_title, wf=wf, t_win=t_win, db=True, t_shift=t_shift, vmin=vmin, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax)
    
    # ref_type="next_win"
    # freq_ref_step=1
    # t_win = 24
    # t_shift = 4
    # scope=4
    # ax2 = plt.subplot(2, 1, 2)
    # coherogram(ax=ax2, wf_title=wf_title, wf=wf, t_win=t_win, ref_type=ref_type, freq_ref_step=freq_ref_step, scope=scope, t_shift=t_shift, sr=sr, xmax=xmax, ymin=ymin, ymax=ymax)
    
    # plt.tight_layout()
    # plt.show()
    
# creating spectogram
if 1==0:
    # get passed in params
    db=True
    t_win = 132
    t_shift = 1
    sr = 128
    show_plot = True
    ymin=0
    ymax=6
    
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # load vod
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)
        wf = wf[0:100000]

    # filename = 'TH21RearwaveformSOAE'
    # mat = scipy.io.loadmat('SOAE Data/' + 'TH21RearwaveformSOAE.mat')
    # wf = np.squeeze(mat['wf'])
    # wf_title = filename

    # filename = "V&D fig 2A, loc=0.1, glob=0, sr=128.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # # load vod
    # with open(filepath + filename, 'rb') as picklefile:
    #     vod = pickle.load(picklefile)
    #     assert isinstance(vod, Vodscillator)
    # wf = vod.SOO_sol
    vmin=-40
    spectrogram(wf=wf, t_win=t_win, t_shift=t_shift, sr=sr, db=db, xmin=0, xmax=100, ymin=ymin, ymax=ymax, vmin=vmin, vmax=30, show_plot=True)
    # coherence_vs_psd(wf=wf, t_win=t_win, t_shift=t_shift, sr=sr, db=db, xmin=0, xmax=6, show_plot=show_plot, do_coherence=False)

# comparing sample rate effect on classic phase coherence
if 1==0:
    t_win = 16
    filename = "V&D fig 2A, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load vod
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    wf = vod.SOO_sol[vod.n_transient:]
    quick(wf, sr=128, t_win=t_win, show_plot=False, fig_num=1, wf_title="128", xmin=0, xmax=10)
    quick(wf, sr=128, t_shift=4, t_win=t_win, show_plot=True, fig_num=2, wf_title="128", xmin=0, xmax=10)

    # filename = "V&D fig 2A, loc=0.1, glob=0, sr=512.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # # load vod
    # with open(filepath + filename, 'rb') as picklefile:
    #     vod = pickle.load(picklefile)
    #     assert isinstance(vod, Vodscillator)

    # wf = vod.SOO_sol[vod.n_transient:]
    # coherence_vs_psd(wf, sr=512, t_win=t_win, show_plot=True, fig_num=2, wf_title="512")

# OLD comparing sample rate effect on classic phase coherence
if 1==0:
    filename = "V&D fig 2A, loc=0, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load vod
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)


    vlodder(vod, "psd_vs_coherence", show_plot=False, fig_num=1, wf_title="128", xmin=0, xmax=10)

    filename = "V&D fig 2A, loc=0, glob=0, sr=512.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load vod
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)


    vlodder(vod, "psd_vs_coherence", show_plot=True, fig_num=2, wf_title="512", xmin=0, xmax=10)

# plotting V&D 3A to get sharp peaks
if 1==0:
    filename = "V&D fig 3A, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    vlodder(vod, "psd", window=-1, xmin=0.5, xmax=5.5, fig_num=1, db=True, wf_title="V&D fig 3A, loc=0.1, glob=0, sr=128")

# comparing V&D sample rate 128 vs 512
if 1==0:
    xmin = 0
    xmax = 5
    t_win=32
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    # vlodder(vod, "superimpose", xmin=0, xmax=5, show_plot=False, fig_num=1)
    quick(vod.SOO_sol[vod.n_transient:], sr=128, t_win=32, show_plot=False, fig_num=1, xmin=xmin, xmax=xmax, wf_title="SR = 128")


    filename = "V&D fig 2A, loc=0.1, glob=0, sr=512.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    quick(vod.SOO_sol[vod.n_transient:], sr=512, t_win=32, fig_num=1, xmin=xmin, xmax=xmax, wf_title="SR = 512")
    # vlodder(vod, "superimpose", xmin=0, xmax=5, show_plot=True, fig_num=2)

# comparing F&B sample rate 128 vs 512
if 1==0:
    xmin = 0
    xmax = 5
    # ymin = -60
    # ymax = 60
    # xmin = None
    # xmax = None
    ymin = None
    ymax = None
    t_win=64
    filename = "F&B fig 2D, noniso, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    quick(vod.SOO_sol[vod.n_transient:], sr=128, t_win=t_win, ymin=ymin, ymax=ymax, show_plot=True, do_coherence=False,fig_num=1, xmin=xmin, xmax=xmax, wf_title="Seth's iso, loc=0.1, glob=0: SR = 128")


    filename = "F&B fig 2D, noniso, loc=0.1, glob=0, sr=512.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    quick(vod.SOO_sol[vod.n_transient:], sr=512, t_win=t_win, fig_num=1, ymin=ymin, ymax=ymax,xmin=xmin, do_coherence=False,xmax=xmax, wf_title="Seth's Non-Iso No Glob: SR = 512")



#psd + coherence of generated data
if 1==0:
    sr = 8
    t = np.arange(0, 10, 1/sr)
    noise_amp = 0
    noise = np.random.uniform(-noise_amp, noise_amp, len(t))
    freqs = [1, 2, 3, 4, 5]
    wf = noise
    for freq in freqs[0:2]:
        wf = wf + np.sin(2*np.pi*freq*t + 2)
    for freq in freqs[1:4]:
        wf = wf + 3*np.cos(2*np.pi*freq*t)

    xmin=None
    xmax=None
    xmin=0
    xmax=10
    quick(wf=wf, sr=sr, t_win=4, xmin=xmin, xmax=xmax, do_psd=True)

#psd + coherence of generated data
if 1==0:
    quick(wf, sr, xmax = 0.1, psd_shift = 0, max_vec_strength=1)

#psd + coherence of soae anolis data
if 1==0:
    # Load the .mat file
    filepath = 'C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\SOAE Data'
    filename = '\\2020.02.21 Anolis\\ACsb18learSOAEwfG4.mat'
    mat = scipy.io.loadmat(filepath + filename)
    soae = np.squeeze(mat['wf'])
    wf_title = 'ACsb18learSOAEwfG4.mat'
    fig_num=1
    win_size=128
    show_plot=True
    max_vec_strength=20
    psd_shift=100
    db=True
    do_psd=True
    do_coherence=True,
    xmin=None
    xmax=None
    ymin=None
    ymax=None
    
    # xmin=1.69
    # xmax=1.76
    # xmin = 0.9
    # xmax = 1.17
    xmin=0
    xmax=4
    # ymin=0
    # ymax=20
    # coherence_vs_psd(soae, win_size=win_size, show_plot=False, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                        #    db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)
    win_size = 2
    fig_num = 2
    quick(soae, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                           db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)
    
    # plt.figure(1)
    # # plt.xlim(left=0, right=30)
    # plt.plot(f, coherence, color='purple')
    # plt.xlabel('Frequency [Hz]')  
    # plt.ylabel(f'Vector Strength [max = {max_vec_strength}]')
    # plt.xlim(left = 1720, right = 1760)
    # plt.figure(2)
    # # plt.xlim(left=0, right=30)
    # plt.xlabel('Frequency [Hz]')  
    # plt.ylabel(f'PSD [dB]')
    # plt.plot(f, psd, color='green')
    # plt.xlim(left = 1720, right = 1760)
    # plt.show()

#psd + coherence of JIrear data
if 1==0:
    # filename = "V&D fig 4, loc=0.1, glob=0, sr=512.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # with open(filepath + filename, 'rb') as picklefile:
    #     vod = pickle.load(picklefile)
    #     assert(isinstance(vod, Vodscillator))
    #     # truncate it
    #     wf = vod.SOO_sol[vod.n_transient:]
        
    # wf_title = "V&D fig 4, loc=0.1, glob=0, sr=512"
    # sr=128
    # khz=False
    # xmin=0
    # xmax=64
    
    
    # filename = "V&D fig 4, loc=0.1, glob=0, sr=128"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # with open(filepath + filename + ".pkl", 'rb') as picklefile:
    #     vod = pickle.load(picklefile)
    #     assert(isinstance(vod, Vodscillator))
    #     # truncate it
    #     wf = vod.SOO_sol[vod.n_transient:]
    
    # Load the .mat file
    filename = 'AC6rearSOAEwfB1'
    mat = scipy.io.loadmat('SOAE Data/' + 'TH21RearwaveformSOAE.mat')
    wf = np.squeeze(mat['wf'])
        # set global
    wf_title = filename
    
    sr=44100
    khz=True
    xmin=0
    xmax=24
    ymin=None
    ymax=None
    ymin=0
    ymax=8
    show_plot=False
    do_psd = True
        
    
    # set alterable params
    t_win=0.1
    t_shift=t_win
    downsample_freq = 1
    
    # optionally downsample
    if 1==0:
        sr = sr / 2
        wf = wf[::2]
    
    # optionally add noise
    if 1==1:
        noise_amp = 0
        wf = wf + (noise_amp*np.random.random_sample(len(wf)) - 1)
    
    
    # plot
    
    plt.figure(2)
    
    ref_type="next_win"
    wf_title1 = wf_title + f" with NWPC (noise_amp = {noise_amp})"
    # wf_title1 = wf_title + f" with PR to Next Win"
    ax1 = plt.subplot(2, 1, 2)
    quick(ax=ax1, downsample_freq=downsample_freq, khz=khz, wf_title=wf_title1, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax, do_psd=do_psd)

    ref_type="next_freq"
    wf_title2 = wf_title + f" with NFPC (noise_amp = {noise_amp})"
    # wf_title2 = wf_title + f" with PR to Higher Freq"
    ax2 = plt.subplot(2, 1, 1)
    quick(ax=ax2, khz=khz, downsample_freq=downsample_freq, wf_title=wf_title2, wf=wf, t_win=t_win, ref_type=ref_type, t_shift=t_shift, sr=sr, xmin=xmin, xmax=xmax, do_psd=do_psd)
    
    plt.tight_layout()
    plt.show()
#psd + coherence of vodscillators
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)
    
    wf = vod.SOO_ss_sol.flatten()
    wf_title = vod.name
    fig_num=1
    win_size=6
    show_plot=True
    max_vec_strength=10
    psd_shift=20
    db=True
    do_psd=True
    do_coherence=True,
    xmin=None
    xmax=None
    ymin=None
    ymax=None
    xmin=0
    xmax=3
    # xmin=1.5
    # xmax=2
    ymin=0
    ymax=30
    # coherence_vs_psd(wf, win_size=win_size, show_plot=False, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                        #    db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

    # win_size = 16
    # fig_num = 2
    quick(wf, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                           db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

#freq cluster of vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)
    # plt.subplot(2, 1, 1)
    # plots.heat_map(v, min_freq=1, max_freq=5)

    # plt.subplot(2, 1, 2)
    # plots.heat_map(v, min_freq=1, max_freq=5, db=False)
    # plt.show()

    # vlodder(vod, "heat_map")
    heat_map(vod, min_freq=0.1, max_freq=6, vmin=40, vmax=None)
    plt.show()

#psd pre or post summing oscillators of vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)
    xmin = 0
    xmax = 10
    plt.subplot(2, 1, 1)
    vlodder(vod, "psd", xmin = xmin, xmax = xmax, show_plot=False)
    plt.subplot(2, 1, 2)
    vlodder(vod, "pre_psd", xmin = xmin, xmax = xmax)

# plot waveform of vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)
    vlodder(vod, "wf", xmin=0, xmax=200)

#freq cluster of F&B vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "F&B fig 2D NEW FREQS.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)
    vlodder(vod, "cluster")


