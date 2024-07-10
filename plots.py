import numpy as np
import matplotlib.pyplot as plt
import pickle
import timeit
from vodscillator import *
from scipy.fft import rfft, rfftfreq

# define helper functions
def get_windowed_fft(wf, sample_rate, t_win, num_wins=None):
  """ Gets the windowed fft of the given waveform with given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
        defaults to 44100 
      t_win: float
        length (in time) of each window
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size

  """
  
  # get sample_spacing
  sample_spacing = 1/sample_rate
  # calculate number of windows (unless it's passed in)
  if not num_wins:
    wf_tf = len(wf) / sample_rate
    num_wins = int(wf_tf / t_win)

  # calculate num_win_pts
  num_win_pts = sample_rate * t_win

  # initialize matrix which will hold the windowed waveform
  windowed_wf = np.zeros((num_wins, num_win_pts))
  for win in range(num_wins):
      win_start = win*num_win_pts
      win_end = (win+1)*num_win_pts
      # grab the (real part of the) waveform in this window
      windowed_wf[win, :] = wf[win_start:win_end].real

  # Now we do the ffts!

  # get frequency axis 
  freq_ax = rfftfreq(num_win_pts, sample_spacing)
  num_freq_pts = len(freq_ax)
  # get fft of each window
  windowed_fft = np.zeros((num_wins, num_freq_pts), dtype=complex)
  for win in range(num_wins):
    windowed_fft[win, :] = rfft(windowed_wf[win, :])
  
  return freq_ax, windowed_fft

def get_psd(wf, sample_rate, t_win, num_wins=None, windowed_fft=None):
  """ Gets the PSD of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate:
        defaults to 44100 
      t_win: float
        length (in time) of each window
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      windowed_fft: any, Optional
        If you want to avoide recalculating the windowed fft, pass it in here!
  """
  # if you passed the windowed_fft in then we'll skip over to the else statement
  if not windowed_fft:
    freq_ax, windowed_fft = get_windowed_fft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins)
  else:
    # ...and we'll calculate the fft_freq manually
    num_win_pts = sample_rate * t_win
    sample_spacing = 1/sample_rate
    freq_ax = rfftfreq(num_win_pts, sample_spacing)
    
  # calculate necessary params from the windowed_fft
  num_wins = np.size(windowed_fft, 0)
  num_freq_pts = np.size(windowed_fft, 1)

  # get num_win_pts
  num_win_pts = sample_rate * t_win

  # initialize array
  windowed_psd = np.zeros((num_wins, num_freq_pts))

  # calculate the normalizing factor (canonical for discrete PSD)
  normalizing_factor = sample_rate * num_win_pts
  # get PSD for each window
  for win in range(num_wins):
    windowed_psd[win, :] = ((np.abs(windowed_fft[win, :]))**2) / normalizing_factor
  # average over all windows
  psd = np.mean(windowed_psd, 0)
  return freq_ax, psd

def get_coherence(wf, sample_rate, t_win, num_wins=None, windowed_fft=None):
  """ Gets the PSD of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate:
        defaults to 44100 
      t_win: float
        length (in time) of each window
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      windowed_fft: any, Optional
        If you want to avoide recalculating the windowed fft, pass it in here!
  """
  # if you passed the windowed_fft in then we'll skip over to the else statement
  if not windowed_fft:
    freq_ax, windowed_fft = get_windowed_fft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins)
  else:
    # ...and we'll calculate the fft_freq manually
    num_win_pts = sample_rate * t_win
    sample_spacing = 1/sample_rate
    freq_ax = rfftfreq(num_win_pts, sample_spacing)

  
  # calculate necessary params from the windowed_fft
  wfft_size = np.shape(windowed_fft)
  num_win = wfft_size[0]
  num_freq_pts = wfft_size[1]

  # get phases
  phases = np.angle(windowed_fft)

  # initialize array for phase diffs
  num_win_pairs = num_win - 1

  phase_diffs = np.zeros((num_win_pairs, num_freq_pts))
  
  for interval in range(0, num_win_pairs):
    # take the difference between the phases in this current window and the next
    phase_diffs[interval] = phases[interval + 1] - phases[interval]

  # get the sin and cos of the phase diffs, and average over the window pairs
  xx= np.mean(np.sin(phase_diffs),axis=0)
  yy= np.mean(np.cos(phase_diffs),axis=0)

  # finally, output the vector strength (for each frequency)
  coherence = np.sqrt(xx**2 + yy**2)

  return freq_ax, coherence

def coherence_vs_psd(wf, sample_rate, t_win, num_wins=None, max_vec_strength=1, psd_shift=0, db=True, xmin=0, xmax=None, 
                     ymin=None, ymax=None, wf_title=None, show_plot=True, do_psd = True, do_coherence = True, fig_num=1):
  """ Plots the power spectral density and phase coherence of an input waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
      t_win: float
        length (in time) of each window
      num_wins: int, Optional
        If this isn't passed, then it will just get the maximum number of windows of the given size
      max_vec_strength: int, Optional
        multiplier on the vector strength of phase coherence; defaults to 1
      psd_shift: int, Optional
        shifts the PSD up or down
      db: bool, Optional
        Chooses whether to plot PSD on a dB (10*log_10) scale
      xmin: float, Optional
        Defaults to 0
      xmax: float, Optional
      ymin: float, Optional
      ymax: float, Optional
      wf_title: String, Optional
        Plot title is: "Phase Coherence and PSD of {wf_title}"
      wf_comp: str, Optional
      show_plot: bool, Optional
        Repress showing plot
      do_psd: bool, Optional
        Repress psd part of plot
      do_coherence: bool, Optional
        Repress coherence part of plot
      fig_num: Any, Optional

  """
  # get windowed_fft so we don't have to do it twice below
  windowed_fft, fft_freqs = get_windowed_fft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins)

  # get PSD
  psd = get_psd(wf, sample_rate, t_win, windowed_fft=windowed_fft)

  # get coherence
  coherence = get_coherence(wf, sample_rate, t_win, windowed_fft=windowed_fft)

  # PLOT!
  f = fft_freqs / 1000
  
  y1 = psd
  if (db == True):
    y1 = 10*np.log10(y1)
  y1 = y1 + psd_shift
  y2 = max_vec_strength*coherence

  plt.figure(fig_num)

  if do_psd:
    # plt.axes().fill_between(f, y1, 0, color='green', alpha=.2)
    plt.plot(f, y1, color = "green", lw=1, label="Power", alpha=0.7)

  if do_coherence:
    plt.plot(f, y2, color = "purple", lw=1, label='Phase Coherence')
  plt.xlabel('Frequency [kHz]')  
  plt.ylabel(f'Power [dB] / Vector Strength [max = {max_vec_strength}]')
  if psd_shift:
    plt.ylabel(f'Power [dB] + {psd_shift} / Vector Strength [max = {max_vec_strength}]')
  plt.legend() 

  # set title
  if wf_title:
    plt.title(f"Phase Coherence and PSD of {wf_title}")
  else:
    plt.title("Phase Coherence and PSD of Waveform")

  # finally, overwrite any default x and y lims (this does nothing if none were inputted)
  plt.xlim(left = xmin, right = xmax)
  plt.ylim(bottom = ymin, top = ymax)
  
  if show_plot:
    plt.show()

  return f, coherence, psd


def phase_portrait(wf, wf_title="Sum of Oscillators"):
    xdot = np.imag(wf)
    x = np.real(wf)
    plt.plot(x, xdot)
    plt.title("Phase Portrait of " + wf_title)
    plt.grid()
    plt.show()

# Vlodder helper functions:
def get_coherence_vod(vod: Vodscillator, osc=-1):
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the interval #)
  # defaults to osc = -1 which is the sum of oscillators
  if osc == -1:
    wf = vod.SOO_fft[:, :]
  else:
    wf = vod.every_fft[osc, :, :]
  
  # get phases
  phases = np.angle(wf)
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

def get_psd_vod(vod: Vodscillator, osc=-1):
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the interval #)
  if osc == -1:
    # if osc = -1 (the default) we want the summed (SOO) response!
    wf = vod.SOO_fft[:, :]
  else:
    wf = vod.every_fft[osc, :, :]

  # take the amplitude squared and normalize
  psd = ((np.abs(wf))**2) / (vod.sample_rate * vod.n_ss)
  # average over windows
  avg_psd = np.mean(psd, 0)
  
  return avg_psd

def vlodder(vod: Vodscillator, plot_type:str, osc=-1, xmin=0, xmax=None, ymin=None, ymax=None, wf_comp="re", 
                    wf_ss=False, show_plot=True, fig_num=1):
  """ Plots various plots from Vodscillator

  Parameters
  ------------
  vod: Vodscillator
    input vodscillator
  plot_type: String
    "coherence" plots phase coherence;
    "cluster" plots V&D style frequency clustering plots;
    "psd" plots power spectral density (if osc=-1, adds up fft of each oscillator and then takes PSD);
    "pre_psd" takes PSD of each oscillator and THEN plots the sum of the PSDs;
    "superimpose" plots phase coherence and PSD;
    "wf" plots a waveform
  
  osc: int, Optional
  xmin: float, Optional
    Defaults to 0
  xmax: float, Optional
  ymin: float, Optional
  ymax: float, Optional
  wf_comp: str, Optional
    Which component of waveform signal to plot: "re" or "im" for real or imaginary, respectively
  wf_ss: boolean, Optional
    If you only want the steady state part of the waveform solution
  show_plot: boolean , Optional
    Gives the ability to suppress the plt.show()
  fig_num: int, Optional
    Only required if plotting multiple figures
    
    """

  # get frequency and time axes
  f = vod.fft_freq
  t = vod.tpoints

  # initialize figure (with fig_num if you're making multiple plots)
  plt.figure(fig_num)

  if plot_type == "superimpose":
    y1 = 10*np.log10(get_psd_vod(vod, osc))
    phase_coherence_max = 10
    y2 = phase_coherence_max*get_coherence_vod(vod, osc)
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

  if plot_type == "psd":
    y = 10*np.log10(get_psd_vod(vod, osc))
    plt.plot(f, y, color = "red", lw=1)
    plt.ylabel('Density')
    plt.xlabel('Frequency')
    # set title
    if osc == -1:
      plt.title("Power Spectral Density of Summed Response")
    else:
      plt.title(f"Power Spectral Density of Oscillator #{osc}")
  
  if plot_type == "pre_psd":
    sum = 0
    for k in range(vod.num_osc):
      sum += get_psd_vod(vod, k)
    y = 10*np.log10(sum)
    plt.plot(f, y, color = "red", lw=1)
    plt.ylabel('Density')
    plt.xlabel('Frequency')
    # set title
    plt.title("Summed Power Spectral Density of Each Oscillator")



  if plot_type == "coherence":
    y = get_coherence_vod(vod, osc)
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
      psd = get_psd_vod(vod, osc)
      # Now, the paper seems to indicate a proper average over each frequency's PSD:
      avg_cluster_freqs[osc] = np.average(vod.fft_freq, weights=psd)
      # But Beth's way was just to use the frequency which has the highest PSD peak
      # avg_cluster_freqs[osc] = vod.fft_freq[np.argmax(psd)]
    
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
  if show_plot:
    plt.show()





  

def heat_map(v=Vodscillator, min_freq=None, max_freq=None, db=True):
  n = v.num_osc
  spectra = (abs(v.every_fft))**2 #first index is oscillator index
  if db:
    spectra = 10*np.log10(spectra)
  avgd_spectra = np.squeeze(np.average(spectra, axis=1)).transpose() #avging over runs
  osc_array = range(0, n)
  freq_array = v.fft_freq
  if min_freq and max_freq:
    # get the index of these frequencies in the frequency array
    min_freq_pt = int(min_freq * v.sample_rate / 2)
    max_freq_pt = int(max_freq * v.sample_rate / 2)
    # restrict to relevant frequencies
    freq_array = freq_array[min_freq_pt:max_freq_pt]
    avgd_spectra = avgd_spectra[min_freq_pt:max_freq_pt,:] 

  xx, yy = np.meshgrid(osc_array, freq_array) 

  if db:
    vmax = 115
  else:
    vmax = 10000000

  plt.pcolormesh(xx, yy, avgd_spectra, cmap='plasma', vmax=vmax)
  label = "PSD"
  if db:
      label = label + " [dB]"
  plt.colorbar(label=label)
  plt.xlabel("Oscillator index")
  plt.ylabel("Frequency (Hz)")
  plt.title("Heat Map of Frequency Clusters")