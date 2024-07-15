from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io

# vod.n_win = vod.n_ss
# vod.save()


# creating coherogram
if 1==1:
    # # good params for next_win!
    # t_win = 4
    # t_shift = 1
    # scope = 5
    
    
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # load wf
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)
        # truncate it
        wf = wf[0:int(len(wf)/10)]
        
    # global
    sample_rate=128
    xmin=None
    xmax=None
    ymin=0
    ymax=6
    
    # important params
    t_win=4
    t_shift=1
    scope=5
    
    # # next_freq
    # ref_type="next_freq"
    # freq_ref_step=10
    # coherogram(wf=wf, t_win=t_win, ref_type=ref_type, freq_ref_step=freq_ref_step, scope=scope, t_shift=t_shift, sample_rate=sample_rate, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=False, fig_num=1)
    # plt.show()
    
    # next_win
    ref_type="next_win"
    coherogram(wf=wf, t_win=t_win, ref_type=ref_type, scope=scope, t_shift=t_shift, sample_rate=sample_rate, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=False, fig_num=2)
    


    vmin=-40
    show_plot = True
    spectrogram(wf=wf, t_win=t_win, db=True, t_shift=t_shift, vmin=vmin, sample_rate=sample_rate, xmax=xmax, ymin=ymin, ymax=ymax, show_plot=show_plot, fig_num=3)

# creating spectogram
if 1==0:
    # get passed in params
    db=True
    t_win = 32
    t_shift = 32
    sample_rate = 128
    show_plot = True
    ymin=0
    ymax=6
    
    filename = "wf - V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Chris's Pickle Jar\\"
    # load vod
    with open(filepath + filename, 'rb') as picklefile:
        wf = pickle.load(picklefile)

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

    spectrogram(wf=wf, t_win=t_win, t_shift=t_shift, sample_rate=sample_rate, db=db, ymin=ymin, ymax=ymax, show_plot=show_plot)

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
    coherence_vs_psd(wf, sample_rate=128, t_win=t_win, show_plot=False, fig_num=1, wf_title="128", xmin=0, xmax=10)
    coherence_vs_psd(wf, sample_rate=128, t_shift=4, t_win=t_win, show_plot=True, fig_num=2, wf_title="128", xmin=0, xmax=10)

    # filename = "V&D fig 2A, loc=0.1, glob=0, sr=512.pkl"
    # filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # # load vod
    # with open(filepath + filename, 'rb') as picklefile:
    #     vod = pickle.load(picklefile)
    #     assert isinstance(vod, Vodscillator)

    # wf = vod.SOO_sol[vod.n_transient:]
    # coherence_vs_psd(wf, sample_rate=512, t_win=t_win, show_plot=True, fig_num=2, wf_title="512")

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
    coherence_vs_psd(vod.SOO_sol[vod.n_transient:], sample_rate=128, t_win=32, show_plot=False, fig_num=1, xmin=xmin, xmax=xmax, wf_title="SR = 128")


    filename = "V&D fig 2A, loc=0.1, glob=0, sr=512.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    coherence_vs_psd(vod.SOO_sol[vod.n_transient:], sample_rate=512, t_win=32, fig_num=1, xmin=xmin, xmax=xmax, wf_title="SR = 512")
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

    coherence_vs_psd(vod.SOO_sol[vod.n_transient:], sample_rate=128, t_win=t_win, ymin=ymin, ymax=ymax, show_plot=True, do_coherence=False,fig_num=1, xmin=xmin, xmax=xmax, wf_title="Seth's iso, loc=0.1, glob=0: SR = 128")


    filename = "F&B fig 2D, noniso, loc=0.1, glob=0, sr=512.pkl"
    filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
    # load apc data
    with open(filepath + filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        assert isinstance(vod, Vodscillator)

    coherence_vs_psd(vod.SOO_sol[vod.n_transient:], sample_rate=512, t_win=t_win, fig_num=1, ymin=ymin, ymax=ymax,xmin=xmin, do_coherence=False,xmax=xmax, wf_title="Seth's Non-Iso No Glob: SR = 512")

# psd + coherence of vodscillators
if 1==0:
    # Open pickled vodscillator
    filename = "V&D fig 2A.pkl"
    with open(filename, 'rb') as picklefile:
        vod = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(vod, Vodscillator)

    max_vec_strength = 20
    xmax = 10
    wf = np.sum(vod.sol[:, vod.n_transient:], 0), vod.sample_rate
    t_win = 64
    coherence_vs_psd(wf, vod.sample_rate, t_win, num_wins=None, xmax = xmax, ymin=0, ymax = 30, max_vec_strength=max_vec_strength, fig_num=1)
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

#psd + coherence of generated data
if 1==0:
    coherence_vs_psd(wf, sr, xmax = 0.1, psd_shift = 0, max_vec_strength=1)

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
    coherence_vs_psd(soae, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
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
    # coherence_vs_psd(soae, win_size=win_size, show_plot=False, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                        #    db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)
    win_size=16
    fig_num = 2
    coherence_vs_psd(soae, win_size=win_size, show_plot=True, max_vec_strength=max_vec_strength,psd_shift=psd_shift, 
                           db=db, wf_title=wf_title, do_psd=do_psd,do_coherence=do_coherence,xmin = xmin, xmax=xmax, ymin=ymin, ymax=ymax, fig_num = fig_num)

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
    vod.do_fft()
    vod.save()
    vlodder(vod, "cluster")

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


    