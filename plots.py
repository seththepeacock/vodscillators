import numpy as np
import matplotlib.pyplot as plt
import pickle
from vodscillators import *
from scipy.fft import rfft, rfftfreq



def coherence_vs_PSD(wf, sample_rate, win_size = 16):
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
    

def phase_portrait(v):
    xdot = np.imag((v.sol))
    x = np.real((v.sol))
    plt.plot(x, xdot)
    plt.grid()


#write stuff like this!!!
#vlodder.coherence(v.SOO_fft)

def heat_map(v):
    pass

def vlodder(vod: Vodscillator, plot_type: str, osc=-1, xmin=0, xmax=None, ymin=None, ymax=None, wf_comp="re", 
                    wf_ss=False, fig_num=1):
  """
  Plots various plots
  Parameters
  ------------
  vod: Vodscillator
    input vodscillator
  plot_type: String
    "coherence" plots phase coherence,
    "cluster" plots V&D style frequency clustering plots,
    "PSD" plots power spectral density,
    "superimpose" plots phase coherence and PSD
    "wf" plots a waveform
  
  osc: int, Optional
  xmin: float, Optional
    Defaults to 0
  xmax: float, Optional
  ymin: float, Optional
    Defaults to 0
  ymax: float, Optional
  wf_comp: str, Optional
    Which component of waveform signal to plot: "re" or "im" for real or imaginary, respectively
  wf_ss: boolean, Optional
    If you only want the steady state part of the waveform solution
  fig_num: int, Optional
    Only required if plotting multiple figures
    
    """
  # Plotter helper functions:
  def get_coherence(osc=-1):
    # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the interval #)
    # defaults to osc = -1 which is the sum of oscillators
    if osc == -1:
      y = vod.SOO_fft[:, :]
    else:
      y = vod.every_fft[osc, :, :]
    
    # get phases
    phases = np.angle(y)
    # initialize array for phase diffs
    phase_diffs = np.zeros((vod.num_intervals - 1, vod.num_freq_points))
    
    for interval in range(0, vod.num_intervals - 1):
      # take the difference between the phases in this current interval and the next
      phase_diffs[interval] = phases[interval + 1] - phases[interval]

    # get the average sin and cos of the phase diffs
    xx= np.mean(np.sin(phase_diffs),axis=0)
    yy= np.mean(np.cos(phase_diffs),axis=0)

    # finally, output the vector strength (for each frequency)
    return np.sqrt(xx**2 + yy**2)

  def get_psd(osc=-1):
    # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the interval #)
    if osc == -1:
      # if osc = -1 (the default) we want the summed (SOO) response!
      y = vod.SOO_fft[:, :]
    else:
      y = vod.every_fft[osc, :, :]

    # take the amplitude squared and normalize
    psd_ = ((np.abs(y))**2) / (vod.sample_rate * vod.n_ss)
    # average over windows
    avg_psd = np.mean(psd_, 0)
    
    return avg_psd

  # get frequency and time axes
  f = vod.fft_freq
  t = vod.tpoints

  # initialize figure (with fig_num if you're making multiple plots)
  plt.figure(fig_num)

  if plot_type == "superimpose":
    y1 = 10*np.log10(get_psd(osc))
    phase_coherence_max = 10
    y2 = phase_coherence_max*get_coherence(osc)
    plt.plot(f, y1, color = "red", lw=1, label="Power")
    plt.plot(f, y2, color = "purple", lw=1, label='Phase Coherence')
    plt.xlabel('Frequency [Hz]')  
    plt.ylabel(f'Power [dB] / Vector Strength [max = {phase_coherence_max}]')
    plt.legend() 

    # set title
    if osc == -1:
      plt.title("Phase Coherence and PSD of Summed Response")
    else:
      plt.title(f"Phase Coherence and PSD of Oscillator #{osc}")

  if plot_type == "PSD":
    y = 10*np.log10(get_psd(osc))
    plt.plot(f, y, color = "red", lw=1)
    plt.ylabel('Density')
    plt.xlabel('Frequency')
    # set title
    if osc == -1:
      plt.title("Power Spectral Density of Summed Response")
    else:
      plt.title(f"Power Spectral Density of Oscillator #{osc}")

  if plot_type == "coherence":
    y = get_coherence(osc)
    plt.plot(f, y1, color = "purple", lw=5)
    plt.xlabel('Frequency [Hz]')  
    plt.ylabel('Power / Vector Strength')

    # set title
    if osc == -1:
      plt.title("Phase Coherence")
    else:
      plt.title(f"Phase Coherenceof Oscillator #{osc}")
    

  if plot_type == "cluster":
  # Creates V&D style frequency clustering plots
  # first, we get our curve of characteristic frequencies
    char_freqs = vod.omegas / (2*np.pi)
    # next, we get our "average position amplitudes" (square root of the average of the square of the real part of z)
    avg_position_amplitudes = np.zeros(vod.num_osc)
    # and the average frequency of each oscillator
    avg_cluster_freqs = np.zeros(vod.num_osc)
    for osc in range(vod.num_osc):
      # get the average amplitude (bottom line on V&D figure)
      avg_position_amplitudes[osc] = np.sqrt(np.mean((vod.ss_sol[osc].real)**2))
      # get the average cluster frequency
      # first get the psd of this oscillator
      psd = get_psd(osc)
      # Now, the paper seems to indicate a proper average over each frequency's PSD:
      # s.avg_cluster_freqs[osc] = np.average(s.fft_freq, weights=psd)
      # But Beth's way was just to use the frequency which has the highest PSD peak
      avg_cluster_freqs[osc] = vod.fft_freq[np.argmax(psd)]
    
    plt.plot(avg_cluster_freqs, '-o', label="Average frequency")
    plt.plot(avg_position_amplitudes, label="Amplitude")
    plt.plot(char_freqs, '--', label="Characteristic frequency")
    plt.ylabel('Average Frequency')
    plt.xlabel('Oscillator Index')
    plt.title(f"Frequency Clustering with Noise Amp: Local = {vod.loc_noise_amp}, Global = {vod.glob_noise_amp}")
    plt.legend()
  
  if plot_type == "wf":
    if osc == -1: #because -1 means "sum"
      y = vod.SOO_sol
      title = "Waveform of Summed Response"
    else:
      y = vod.sol[osc]
      title = f"Waveform of Oscillator #{osc}"

    if wf_comp == "im":
      y = y.imag
      title = "Velocity " + title
      plt.ylabel("Velocity")
    elif wf_comp == "re":
      y = y.real
      title = "Position " + title
      plt.ylabel("Position")

    if wf_ss:
      t = t[vod.n_transient:]
      y = y[vod.n_transient:]
      title = "Steady State " + title

    plt.plot(t, y)
    plt.xlabel("Time")
    plt.title(title)
    
  # finally, overwrite any default x and y lims (this does nothing if none were inputted)
  plt.xlim(left = xmin, right = xmax)
  plt.ylim(bottom = ymin, top = ymax)
  # and show plot!
  plt.show()





  

  
  
