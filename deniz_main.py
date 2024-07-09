from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from plots import *
import scipy.io

# Open APC and plot
if 1==1:
    cluster_width=0.01
    delta_f=0.001
    num_t_wins=100
    t_win_size=1/2
    amp_weights=True
    f_min=1
    f_max=5

    #open our stuff
    filename = f"cluster_width={cluster_width}, delta_f={delta_f}, num_t_wins={num_t_wins}, t_win_size={t_win_size}, amp_weights={amp_weights}.pkl"
    filepath = "/home/deniz/Dropbox/vodscillators/"
    with open(filepath + filename, 'rb') as picklefile:
        p = pickle.load(picklefile)
    vod_file= "/home/deniz/Dropbox/vodscillators/F&B fig 2D iso.pkl"
    with open(vod_file, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)

    # get freqs for new phase coherence
    apc_freqs = np.arange(f_min, f_max, delta_f)

    # get 2 axes for double y axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plot
    ax1.plot(vod.fft_freq, 10*np.log10(get_psd_vod(vod)), label="PSD", color='r')
    ax2.plot(apc_freqs, p, label="APC", color='b')
    # ax2.plot(v.fft_freq, (get_coherence_vod(v)), label="Coherence", color='purple')
    # set labels
    ax1.set_xlabel('Freq')
    ax1.set_ylabel('PSD [dB]', color='r')
    ax2.set_ylabel('Phase Coherence', color='b')
    # set title, show legend, set xlims
    plt.title(f"APC: cluster_width={cluster_width}, delta_f={delta_f}, num_t_wins={num_t_wins}, t_win_size={t_win_size}, amp_weights={amp_weights}")
    ax1.legend()
    ax1.set_ylim(-25, 20)
    ax2.legend()
    #ax1.set_ylim(-10, 30)
    plt.xlim(f_min, f_max)
    plt.show()

# Generate and save APC data for vodscillator
if 1==0:
    # calculates APC and then save to file
    def apc_and_save(vod=Vodscillator, cluster_width=float, f_min=float, f_max=float, delta_f=float, num_t_wins=float, t_win_size=float, amp_weights=bool):
        # calculate the apc, and it'll be (temporarily) saved to the vod object
        vod.analytic_phase_coherence(cluster_width=cluster_width, f_min=f_min, f_max=f_max, delta_f=delta_f, num_t_wins=num_t_wins, t_win_size=t_win_size, amp_weights=amp_weights)
        # pickle the apc into its own file
        with open(f"cluster_width={cluster_width}, delta_f={delta_f}, num_t_wins={num_t_wins}, t_win_size={t_win_size}, amp_weights={amp_weights}.pkl", 'wb') as outp:  # Overwrites any existing file with this filename!.
            pickle.dump(vod.apc, outp, pickle.HIGHEST_PROTOCOL)

    # open up a vod
    filepath = "/home/deniz/Dropbox/vodscillators/"
    filename = "F&B fig 2D iso.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)

    # any new vod pickles will have this predefined, but you'll have to calculate it by hand for now!
    vod.t_transient = vod.n_transient / vod.sample_rate

    # define your parameters
    cluster_width=0.01
    f_min=1
    f_max=5
    delta_f=0.001
    num_t_wins=100
    # t_win_size has to be s.t. t_win_size * vod.sample_rate(=128) is an integer!!!
    t_win_size=1/2
    amp_weights=True

    # run fx to get apc and save to a lil pickle
    apc_and_save(vod=vod, cluster_width=cluster_width, f_min=f_min, f_max=f_max, delta_f=delta_f, num_t_wins=num_t_wins, t_win_size=t_win_size, amp_weights=amp_weights)
        
    # change any parameters you want and rerun (note you can copy and paste the same list of args)
    amp_weights=False
    apc_and_save(vod=vod, cluster_width=cluster_width, f_min=f_min, f_max=f_max, delta_f=delta_f, num_t_wins=num_t_wins, t_win_size=t_win_size, amp_weights=amp_weights)
    
    # let's chang some more params!
    amp_weights=True
    t_win_size=1/16
    apc_and_save(vod=vod, cluster_width=cluster_width, f_min=f_min, f_max=f_max, delta_f=delta_f, num_t_wins=num_t_wins, t_win_size=t_win_size, amp_weights=amp_weights)
   
# psd + coherence of vodscillators with 4 window sizes
if 1==0:
    # Open pickled vodscillator
    filename = "V&D fig 2A.pkl"
    with open(filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)

    max_vec_strength = 20
    xmax = 10
    coherence_vs_psd(np.sum(vod.ss_sol, 0), vod.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=8, max_vec_strength=max_vec_strength, fig_num=1)
    coherence_vs_psd(np.sum(vod.ss_sol, 0), vod.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=16, max_vec_strength=max_vec_strength, fig_num=2)
    coherence_vs_psd(np.sum(vod.ss_sol, 0), vod.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=32, max_vec_strength=max_vec_strength, fig_num=3)
    coherence_vs_psd(np.sum(vod.ss_sol, 0), vod.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=40, max_vec_strength=max_vec_strength, fig_num=4)
    plt.show()

#psd + coherence of generated data
if 1==0:
    sr = 512
    t = np.arange(0, 1000, 1/sr)
    noise_amp = 50
    noise = np.random.uniform(-noise_amp, noise_amp, len(t))
    freqs = [1, 2, 3, 4, 5]
    wf = noise
    for freq in freqs:
        wf = wf + np.sin(2*np.pi*freq*t)
    coherence_vs_psd(wf, sr, xmax = 0.1, psd_shift = 0, max_vec_strength=1)


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
    coherence_vs_psd(wf, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                           db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

#freq cluster of vodscillator
if 1==0:


    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.1.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)
    # plt.subplot(2, 1, 1)
    # plots.heat_map(v, min_freq=1, max_freq=5)

    # plt.subplot(2, 1, 2)
    # plots.heat_map(v, min_freq=1, max_freq=5, db=False)
    # plt.show()
    vlodder(vod, "cluster")

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

