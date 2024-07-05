from vodscillators import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
import plots
import scipy.io


if 1 == 0:
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

if 1 == 0:
    sr = 512
    t = np.arange(0, 1000, 1/sr)
    noise_amp = 50
    noise = np.random.uniform(-noise_amp, noise_amp, len(t))
    freqs = [1, 2, 3, 4, 5]
    wf = noise
    # for freq in freqs:
    #     wf = wf + np.sin(2*np.pi*freq*t)
    plots.coherence_vs_PSD(wf, sr, xmax = 100, psd_shift = 0, max_vec_strength=1)

if 1 == 1:
    # Load the .mat file
    mat = scipy.io.loadmat('SOAE Data/fwavef.JIrearSOAEwf2.CF1723.BW30.mat')
    soae = mat['fwavef'][0]
    max_vec_strength = 10
    psd_shift = 105
    plots.coherence_vs_PSD(soae, win_size=1, make_plot=True, xmin=1720, xmax=1760, max_vec_strength=max_vec_strength, psd_shift=psd_shift)
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

if 1==0:
    # Open pickled vodscillator
    filename = "V&D fig 2A.pkl"
    with open(filename, 'rb') as picklefile:
        v = pickle.load(picklefile)
        # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
        assert isinstance(v, Vodscillator)


