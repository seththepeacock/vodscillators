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
    f_min=1
    f_max=5
    delta_f=0.001
    duration=50
    t_win_size=1/2
    amp_weights=True

    #open our stuff
    filename = f"cluster_width={cluster_width}, delta_f={delta_f}, duration={duration}, t_win_size={t_win_size}, amp_weights={amp_weights}.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\APC V&D fig 2A, loc=0.1, glob=0\\"
    with open(filepath + filename, 'rb') as picklefile:
        p = pickle.load(picklefile)
    vod_file= "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(vod_file, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)

    # get freqs for new phase coherence
    apc_freqs = np.arange(f_min, f_max, delta_f)

    # get 2 axes for double y axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plot
    ax1.plot(v.fft_freq, 10*np.log10(get_psd_vod(v)), label="PSD", color='r')
    ax2.plot(apc_freqs, p, label="APC")
    ax2.plot(v.fft_freq, (get_coherence_vod(v)), label="Coherence")
    # set labels
    ax1.set_xlabel('Freq')
    ax1.set_ylabel('PSD [dB]', color='r')
    ax2.set_ylabel('Phase Coherence')
    # set title, show legend, set xlims
    plt.title(f"APC: cluster_width={cluster_width}, delta_f={delta_f}, duration={duration}, t_win_size={t_win_size}, amp_weights={amp_weights}")
    ax1.legend()
    ax2.legend()
    plt.xlim(f_min, f_max)
    plt.show()
# Generate and save APC data for vodscillator
if 1==0:
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)
    start = timeit.default_timer()
    v.t_transient = v.n_transient / v.sample_rate
    cluster_width=0.01
    f_min=1
    f_max=5
    delta_f=0.001
    duration=50
    t_win_size=1/2
    amp_weights=True
    v.analytic_phase_coherence(cluster_width=cluster_width, f_min=f_min, f_max=f_max, delta_f=delta_f, duration=duration, t_win_size=t_win_size, amp_weights=amp_weights)
    ""
    with open(f"cluster_width={cluster_width}, delta_f={delta_f}, duration={duration}, t_win_size={t_win_size}, amp_weights={amp_weights}.pkl", 'wb') as outp:  # Overwrites any existing file with this filename!.
        pickle.dump(v.apc, outp, pickle.HIGHEST_PROTOCOL)
    
    amp_weights=False
    with open(f"cluster_width={cluster_width}, delta_f={delta_f}, duration={duration}, t_win_size={t_win_size}, amp_weights={amp_weights}.pkl", 'wb') as outp:  # Overwrites any existing file with this filename!.
        pickle.dump(v.apc, outp, pickle.HIGHEST_PROTOCOL)
    
    amp_weights=True
    t_win_size=1/16
    with open(f"cluster_width={cluster_width}, delta_f={delta_f}, duration={duration}, t_win_size={t_win_size}, amp_weights={amp_weights}.pkl", 'wb') as outp:  # Overwrites any existing file with this filename!.
        pickle.dump(v.apc, outp, pickle.HIGHEST_PROTOCOL)

    plt.plot(v.apc_freqs, v.apc)
    stop = timeit.default_timer()
    
    print(f"Whew... that took {stop-start} seconds, {(stop-start)/60} minutes!")
    plt.show()
    

# psd + coherence of vodscillators with 4 window sizes
if 1==0:
    # Open pickled vodscillator
    filename = "V&D fig 2A.pkl"
    with open(filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)

    max_vec_strength = 20
    xmax = 10
    plots.coherence_vs_PSD(np.sum(v.ss_sol, 0), v.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=8, max_vec_strength=max_vec_strength, fig_num=1)
    plots.coherence_vs_PSD(np.sum(v.ss_sol, 0), v.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=16, max_vec_strength=max_vec_strength, fig_num=2)
    plots.coherence_vs_PSD(np.sum(v.ss_sol, 0), v.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=32, max_vec_strength=max_vec_strength, fig_num=3)
    plots.coherence_vs_PSD(np.sum(v.ss_sol, 0), v.sample_rate, xmax = xmax, ymin=0, ymax = 30, win_size=40, max_vec_strength=max_vec_strength, fig_num=4)
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
    plots.coherence_vs_PSD(wf, sr, xmax = 0.1, psd_shift = 0, max_vec_strength=1)

#psd + coherence of generated data
if 1==0:
    plots.coherence_vs_PSD(wf, sr, xmax = 0.1, psd_shift = 0, max_vec_strength=1)

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
    # plots.coherence_vs_PSD(soae, win_size=win_size, show_plot=False, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                        #    db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)
    win_size = 2
    fig_num = 2
    plots.coherence_vs_PSD(soae, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
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
    # Load the .mat file
    filename = 'TH21RearwaveformSOAE'
    mat = scipy.io.loadmat('SOAE Data/' + 'TH21RearwaveformSOAE.mat')
    soae = np.squeeze(mat['wf'])
    wf_title = filename
    fig_num=1
    win_size=128
    show_plot=True
    max_vec_strength=100
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
    xmin=1
    xmax=5
    # ymin=0
    # ymax=1
    # plots.coherence_vs_PSD(soae, win_size=win_size, show_plot=False, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                        #    db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)
    win_size=16
    fig_num = 2
    plots.coherence_vs_PSD(soae, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                           db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

#psd + coherence of vodscillators
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)
    
    wf = v.SOO_ss_sol.flatten()
    wf_title = v.name
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
    # plots.coherence_vs_PSD(wf, win_size=win_size, show_plot=False, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                        #    db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

    # win_size = 16
    # fig_num = 2
    plots.coherence_vs_PSD(wf, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                           db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

#freq cluster of vodscillator
if 1==0:


    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.1.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)
    # plt.subplot(2, 1, 1)
    # plots.heat_map(v, min_freq=1, max_freq=5)

    # plt.subplot(2, 1, 2)
    # plots.heat_map(v, min_freq=1, max_freq=5, db=False)
    # plt.show()
    plots.vlodder(v, "cluster")

#psd pre or post summing oscillators of vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)
    xmin = 0
    xmax = 10
    plt.subplot(2, 1, 1)
    plots.vlodder(v, "psd", xmin = xmin, xmax = xmax, show_plot=False)
    plt.subplot(2, 1, 2)
    plots.vlodder(v, "pre_psd", xmin = xmin, xmax = xmax)

# plot waveform of vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "V&D fig 2A, loc=0.1, glob=0.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)
    plots.vlodder(v, "wf", xmin=0, xmax=200)

#freq cluster of F&B vodscillator
if 1==0:
    # Open pickled vodscillator
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    filename = "F&B fig 2D NEW FREQS.pkl"
    with open(filepath + filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)
    vlodder(v, "cluster")


    