import numpy as np
import matplotlib.pyplot as plt
from vodscillator import *
from scipy.fft import rfft, rfftfreq


# define helper functions

def NEW_get_windowed_fft(wf, sample_rate, t_win, t_shift=None, num_wins=None):
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
  # if you didn't pass in t_shift we'll assume you want no overlap - each new window starts at the end of the last!
  if t_shift is None:
    t_shift=t_win
  
  
  win_starts = np.arange(0, stop, t_shift)

  # if number of windows is passed in, we make sure it's less than the length of win_starts
  if num_wins is not None:
    if num_wins > len(win_starts):
      raise Exception("That's more windows than we can manage! Decrease num_wins!")
  else:
    # if no num_wins is passed in, we'll just use the max number of windows
    num_wins = len(win_starts)







  # get sample_spacing
  sample_spacing = 1/sample_rate

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

def get_windowed_fft(wf, sample_rate, t_win, t_shift=None, num_wins=None):
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
  # if you didn't pass in t_shift we'll assume you want no overlap - each new window starts at the end of the last!
  if t_shift is None:
    t_shift=t_win

  # calculate number of windows (unless it's passed in)
  if num_wins is None:
    wf_tf = len(wf) / sample_rate
    num_wins = int(wf_tf / t_win)

  # get sample_spacing
  sample_spacing = 1/sample_rate

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
  if windowed_fft is None:
    freq_ax, windowed_fft = get_windowed_fft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins)
  else:
    # ...and we'll calculate the fft_freq manually
    num_win_pts = sample_rate * t_win
    sample_spacing = 1/sample_rate
    freq_ax = rfftfreq(num_win_pts, sample_spacing)
    
  # calculate necessary params from the windowed_fft
  wfft_size = np.shape(windowed_fft)
  num_wins = wfft_size[0]
  num_freq_pts = wfft_size[1]

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
  if windowed_fft is None:
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
                     ymin=None, ymax=None, wf_title=None, show_plot=True, do_coherence=True, do_psd=True, fig_num=1):
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
        Sets the PSD axis
      ymax: float, Optional
        Sets the PSD axis
      wf_title: String, Optional
        Plot title is: "Phase Coherence and PSD of {wf_title}"
      wf_comp: str, Optional
      do_coherence: bool, Optional
        optionally suppress coherence plot
      do_psd: bool, Optional
        optionally suppress PSD plot
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional

  """
  # get windowed_fft so we don't have to do it twice below
  fft_freqs, windowed_fft = get_windowed_fft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins)

  # get PSD
  psd = get_psd(wf, sample_rate, t_win, windowed_fft=windowed_fft)[1]

  # get coherence
  coherence = get_coherence(wf, sample_rate, t_win, windowed_fft=windowed_fft)[1]

  # PLOT!
  f = fft_freqs
  
  if (db == True):
    psd = 20*np.log10(psd)
  psd = psd + psd_shift
  coherence = max_vec_strength*coherence

  plt.figure(fig_num)

  # get 2 axes for double y axis
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()

  # plot
  if do_coherence:
    ax1.plot(f, coherence, label=f"Coherence: t_win={t_win}", color='purple')
  if do_psd:
    ax2.plot(f, psd, label="PSD", color='r')

  # set labels
  ax1.set_ylabel('Phase Coherence', color='purple')
  ax2.set_xlabel('Freq')
  ax2.set_ylabel('PSD [dB]', color='r')
  ax1.legend()
  ax2.legend()

  # set title
  if wf_title:
    plt.title(f"Phase Coherence and PSD of {wf_title}")
  else:
    plt.title("Phase Coherence and PSD of Waveform")

  # finally, overwrite any default x and y lims (this does nothing if none were inputted)
  plt.xlim(left = xmin, right = xmax)
  ax2.set_ylim(bottom = ymin, top = ymax)
  
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

