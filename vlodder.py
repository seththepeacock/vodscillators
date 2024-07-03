import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq


def coherence(s, input):
    #implement a general coherence 
    print()

  
def plot_waveform(s, osc = -1, component = "re", interval = -1, 
                ss = False, xmin = -0.1, xmax = None, ymin = 0.0, ymax = None, fig_num = 1):
    """ Plots a waveform for a given oscillator

    Parameters
    ------------
        osc: int, Optional
            The index of your oscillator (-1 gives summed response)
        component: str, Optional
            Which component of waveform signal to plot; "re" or "im" for real or imaginary, respectively
        ss: boolean, Optional
            If you only want the steady state part of the solution
        xmin: float, Optional
        xmax: float, Optional
        ymin: float, Optional
        ymax: float, Optional
        fig_num: int, Optional
            Only required if plotting multiple figures

        
    """

    if osc == -1: #because -1 means "sum"
        y = s.summed_sol
    else:
        y = s.sol[osc]

    if component == "im":
        y = y.imag
    elif component == "re":
        y = y.real

    t = s.tpoints

    if ss:
        t = t[s.n_transient:]
        y = y[s.n_transient:]

    fig5 = plt.figure()
    plt.plot(t, y)


def psd(s, osc=-1, interval=-1):
    if osc == -1:
        if interval == -1:  
        y = s.SOO_AOI_fft
        else:
        y = s.SOO_fft[interval]
    else:
        if interval == -1:
        y = s.AOI_fft[osc]
        else:
        y = s.every_fft[osc, interval]

        # square the amplitude
        y = (np.abs(y))**2

        # normalize
        y = y / (s.sample_rate * s.n_ss)
        
    s.psd = y


def plot(s, plot_type="", osc=-1, interval=-1, fig_num=1, xmin = 0, xmax = None, ymin = 0, ymax = None):
    """
    Creates V&D style frequency clustering plots
    Parameters
    ------------
    plot_type: list
        "coherence" plots phase coherence,
        "cluster" plots V&D style frequency clustering plots,
        "PSD" plots power spectral density,
        "superimpose" plots phase coherence and PSD

    fig_num: int, Optional
        Only required if plotting multiple figures

    interval: int, Optional
        Which SS interval to display PSD for, defaults to -1 for average

    xmin: float, Optional
        Defaults to 0
    xmax: float, Optional
    ymin: float, Optional
        Defaults to 0
    ymax: float, Optional

    """

freq = s.fft_freq

if plot_type == "superimpose":
    f = s.fft_freq
    if osc == -1:
    if interval == -1:  
        y = s.SOO_AOI_fft
    else:
        y = s.SOO_fft[interval]
    else:
    if interval == -1:
        y = s.AOI_fft[osc]
    else:
        y = s.every_fft[osc, interval]
    # square the amplitude
    y = (np.abs(y))**2
    # normalize
    y = y / (s.sample_rate * s.n_ss)

    fig1 = plt.figure()
    plt.plot(f, y, color = "red", label="Power")
    plt.plot(freq, s.SOO_phase_coherence * 10, color = "green", lw=1,label='Phase Coherence')
    plt.xlabel('Frequency [Hz]')  
    plt.ylabel('Power / Vector Strength x 10') 
    plt.title("Phase Coherence and PSD of Summed Response") 
    plt.xlim(left = 0, right = 10)
    plt.xlim(left = xmin, right = xmax)
    plt.ylim(bottom = ymin, top = ymax)
    plt.legend()


if plot_type == "coherence":
    fig2 = plt.figure()
    plt.plot(freq/1000,s.coherence,'b-',lw=1,label='X')
    plt.xlabel('Frequency [kHz]')  
    plt.ylabel('Phase Coherence (i.e. vector strength)') 
    plt.title("coherence") 
    plt.xlim([0, 0.1])
    

if plot_type == "cluster":
# first, we get our curve of characteristic frequencies
    s.char_freqs = s.omegas / (2*np.pi)
    # next, we get our "average position amplitudes" (square root of the average of the square of the real part of z)
    s.avg_position_amplitudes = np.zeros(s.num_osc)
    # and the average frequency of each oscillator
    s.avg_cluster_freqs = np.zeros(s.num_osc)
    for osc in range(s.num_osc):
    s.avg_position_amplitudes[osc] = np.sqrt(np.mean((s.ss_sol[osc].real)**2))
    # This is what it seems like they do in the paper:
    #s.avg_cluster_freqs[osc] = np.average(s.fft_freq, weights = np.abs(s.AOI_fft[osc]))
    # This is Beth's way:
    s.avg_cluster_freqs[osc] = s.fft_freq[np.argmax(np.abs(s.AOI_fft[osc]))]
    # now plot!
    
    fig3 = plt.figure()
    plt.plot(s.avg_cluster_freqs, '-o', label="Average frequency")
    plt.plot(s.avg_position_amplitudes, label="Amplitude")
    plt.plot(s.char_freqs, '--', label="Characteristic frequency")
    plt.ylabel('Average Frequency')
    plt.xlabel('Oscillator Index')
    plt.title(f"Frequency Clustering with Noise Amp: Local = {s.loc_noise_amp}, Global = {s.glob_noise_amp}")
    plt.legend()

if plot_type == "PSD":
    fig4 = plt.figure()
    s.psd()
    f = s.fft_freq
    y = s.psd
    #plt.figure(fig_num)
    plt.plot(f, y)
    plt.xlim(left = xmin)
    plt.xlim(right = xmax)
    plt.ylim(bottom = ymin)
    plt.ylim(top = ymax)
    plt.title("Power Spectral Density")
    plt.ylabel('Density')
    plt.xlabel('Frequency')











