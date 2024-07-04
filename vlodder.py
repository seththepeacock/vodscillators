import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.fft import rfft, rfftfreq


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
    # get frequency axis 
    freq_pts = rfftfreq(num_win_pts, sample_spacing)
    # calculate number of windows 
    num_win = np.floor(len(wf) / 16)
    # initialize matrix which will hold the windowed waveform
    windowed_wf = np.zeros(num_win, num_win_pts)
    for k in num_win:
        windowed_wf[k, :] = wf[k * num_win_pts:(k+1)*num_win_pts]
    

    
    
