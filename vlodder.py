import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.fft import rfft, rfftfreq


def plot_coherence_vs_PSD(s, input, window_size, pre-fft = False):
    """
    Plots the power spectral density of an input waveform (or fourier transformed waveform)
    """

    # first, we check if the input is a waveform or fft
    if input_type == "wf":
        input = rfft(input)
        freq_axis = 
    #implement a general coherence 
    print()
