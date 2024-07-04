import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.fft import rfft, rfftfreq
from vodscillators import *


def coherence_vs_PSD(s, wf, sample_rate, win_size = 16):
    """ Plots the power spectral density and phase coherence of an input waveform
    
    Parameters
    ------------
        waveform: array
        window_size: int, Optional
            The # points in each window will be 512 * window_size
            Defaults to 16, which gives a total window size of 8192 which is the standard Vodscillator averaging window
    """
    # get length and spacing of window
    num_win_pts = win_size * 512
    sample_spacing = 1/sample_rate
    # calculate number of windows 
    num_win = np.floor(len(wf) / 16)
    # initialize matrix which will hold the windowed waveform
    windowed_wf = np.zeros(num_win, num_win_pts)
    for k in num_win:
        win_start = k*num_win_pts
        win_end = (k+1)*num_win_pts
        windowed_wf[k, :] = wf[win_start:win_end]

    # Now we do the ffts!

    # get frequency axis 
    freq_pts = rfftfreq(num_win_pts, sample_spacing)
    windowed_fft = np.zeros(num_win, len(freq_pts))
    for k in num_win:
        windowed_fft[k, :] = rfft(windowed_wf[k, :])
    
    # first, get PSD, averaged over all windows
    # average over all windows


    y = (np.abs(y))**2

    # normalize
    y = y / (s.sample_rate * s.n_ss)
      
    s.psd = y
    


def phase_portrait(v):
    xdot = np.imag((v.sol))
    x = np.real((v.sol))
    plt.plot(x, xdot)
    plt.grid()

