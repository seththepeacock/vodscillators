import numpy as np
import matplotlib.pyplot as plt
from vodscillator import *
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

# define helper functions
def get_wfft(wf, sample_rate, t_win, t_shift=None, num_wins=None, return_all=False):
  """ Gets the windowed fft of the given waveform with given window size and given t_shift

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
        defaults to 44100 
      t_win: float
        length (in time) of each window
      t_shift: float
        length (in time) between the start of successive windows
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      return_all: bool, Optional
        Defaults to only returning the wfft; if this is enabled, then a dictionary is returned with keys:
        "wfft", "freq_ax", "win_start_indices"

  """


  # if you didn't pass in t_shift we'll assume you want no overlap - each new window starts at the end of the last!
  if t_shift is None:
    t_shift=t_win
  
  # calculate the number of samples in the window
    # + 1 is because if you have SR=2 and you want a two second window, this will take 5 samples!
  n_win = int(t_win*sample_rate) + 1

  # and the number of samples to shift
    # no + 1 here; if you want to shift it over one second and SR=2, that will be two samples
  n_shift = int(t_shift*sample_rate)

  # get sample_spacing
  sample_spacing = 1/sample_rate

  # first, get the last index of the waveform
  final_wf_index = len(wf) - 1
    # - 1 is because of zero-indexing!
  # next, we get what we would be the largest potential win_start_index
  final_win_start_index = final_wf_index - (n_win-1)
    # start at the final_wf_index. we need to collect n_win points. this final index is our first one, and then we need n_win - 1 more. 
    # So we march back n_win-1 points, and then THAT is the last_potential_win_start_index!
  win_start_indices = np.arange(0, final_win_start_index + 1, n_shift)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_win_start_index!

  # if number of windows is passed in, we make sure it's less than the length of win_start_indices
  if num_wins is not None:
    if num_wins > len(win_start_indices):
      raise Exception("That's more windows than we can manage! Decrease num_wins!")
  else:
    # if no num_wins is passed in, we'll just use the max number of windows
    num_wins = len(win_start_indices)

  windowed_wf = np.zeros((num_wins, n_win))

  for k in range(num_wins):
    win_start = win_start_indices[k]
    win_end = win_start + n_win
    # grab the (real part of the) waveform in this window
    windowed_wf[k, :] = wf[win_start:win_end].real
    # note this grabs the wf at indices win_start, win_start+1, ..., win_end-1
      # if there are 4 samples and t_win=t_shift=1 and SR=1, then n_win=2, n_shift=1 and
      # Thus the first window will be samples 0 and 1, the next 1 and 2...

  # Now we do the ffts!

  # get frequency axis 
  freq_ax = rfftfreq(n_win, sample_spacing)
  num_freq_pts = len(freq_ax)
  # get fft of each window
  wfft = np.zeros((num_wins, num_freq_pts), dtype=complex)
  for k in range(num_wins):
    wfft[k, :] = rfft(windowed_wf[k, :])
  
  if not return_all:
    return wfft
  else: 
    return {  
      "wfft" : wfft,
      "freq_ax" : freq_ax,
      "win_start_indices" : win_start_indices
      }


def get_wfft2(wf, sample_rate, t_win, t_shift=None, num_wins=None, return_all=False):
  """ Gets the windowed fft of the given waveform with given window size and given t_shift

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
        defaults to 44100 
      t_win: float
        length (in time) of each window
      t_shift: float
        length (in time) between the start of successive windows
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      return_all: bool, Optional
        Defaults to only returning the wfft; if this is enabled, then a dictionary is returned with keys:
        "wfft", "freq_ax", "win_start_indices"

  """


  # if you didn't pass in t_shift we'll assume you want no overlap - each new window starts at the end of the last!
  if t_shift is None:
    t_shift=t_win
  
  # calculate the number of samples in the window
    # + 1 is because if you have SR=2 and you want a two second window, this will take 5 samples!
  n_win = int(t_win*sample_rate) + 1

  # and the number of samples to shift
    # no + 1 here; if you want to shift it over one second and SR=2, that will be two samples
  n_shift = int(t_shift*sample_rate)

  # get sample_spacing
  sample_spacing = 1/sample_rate

  # first, get the last index of the waveform
  final_wf_index = len(wf) - 1
    # - 1 is because of zero-indexing!
  # next, we get what we would be the largest potential win_start_index
  final_win_start_index = final_wf_index - (n_win-1)
    # start at the final_wf_index. we need to collect n_win points. this final index is our first one, and then we need n_win - 1 more. 
    # So we march back n_win-1 points, and then THAT is the last_potential_win_start_index!
  win_start_indices = np.arange(0, final_win_start_index + 1, n_shift)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_win_start_index!

  # if number of windows is passed in, we make sure it's less than the length of win_start_indices
  if num_wins is not None:
    if num_wins > len(win_start_indices):
      raise Exception("That's more windows than we can manage! Decrease num_wins!")
  else:
    # if no num_wins is passed in, we'll just use the max number of windows
    num_wins = len(win_start_indices)

  windowed_wf = np.zeros((num_wins, n_win))

  for k in range(num_wins):
    win_start = win_start_indices[k]
    win_end = win_start + n_win
    # grab the (real part of the) waveform in this window
    windowed_wf[k, :] = wf[win_start:win_end].real
    # note this grabs the wf at indices win_start, win_start+1, ..., win_end-1
      # if there are 4 samples and t_win=t_shift=1 and SR=1, then n_win=2, n_shift=1 and
      # Thus the first window will be samples 0 and 1, the next 1 and 2...

  # Now we do the ffts!

  # get frequency axis 
  freq_ax = rfftfreq(n_win, sample_spacing)
  num_freq_pts = len(freq_ax)
  # get fft of each window
  wfft = np.zeros((num_wins, num_freq_pts), dtype=complex)
  for k in range(num_wins):
    wfft[k, :] = rfft(windowed_wf[k, :]*np.hanning(n_win))
  
  if not return_all:
    return wfft
  else: 
    return {  
      "wfft" : wfft,
      "freq_ax" : freq_ax,
      "win_start_indices" : win_start_indices
      }


def get_psd(wf, sample_rate, t_win, num_wins=None, wfft=None, return_all=False):
  """ Gets the PSD of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate:
        defaults to 44100 
      t_win: float
        length (in time) of each window; used in get_wfft and to calculate normalizing factor
      num_wins: int, Optional
        Used in get_wfft;
          if this isn't passed, then just gets the maximum number of windows of the given size
      wfft: any, Optional
        If you want to avoide recalculating the windowed fft, pass it in here!
      return_all: bool, Optional
        Defaults to only returning the PSD averaged over all windows; if this is enabled, then a dictionary is returned with keys:
        "avg_psd", "freq_ax", "win_psd"
  """
  # if you passed the wfft in then we'll skip over this
  if wfft is None:
    wfft = get_wfft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins)

  # we'll calculate the fft_freq manually
  num_win_pts = sample_rate * t_win
  sample_spacing = 1/sample_rate
  freq_ax = rfftfreq(num_win_pts, sample_spacing)
    
  # calculate necessary params from the wfft
  wfft_size = np.shape(wfft)
  num_wins = wfft_size[0]
  num_freq_pts = wfft_size[1]

  # get num_win_pts
  num_win_pts = sample_rate * t_win

  # initialize array
  win_psd = np.zeros((num_wins, num_freq_pts))

  # calculate the normalizing factor (canonical for discrete PSD)
  normalizing_factor = sample_rate * num_win_pts
  # get PSD for each window
  for win in range(num_wins):
    win_psd[win, :] = ((np.abs(wfft[win, :]))**2) / normalizing_factor
  # average over all windows
  avg_psd = np.mean(win_psd, 0)
  if not return_all:
    return avg_psd
  else:
    return {  
      "avg_psd" : avg_psd,
      "freq_ax" : freq_ax,
      "win_psd" : win_psd
      }

def get_coherence(wf, sample_rate, t_win=16, t_shift=1, num_wins=None, wfft=None, return_all=False):
  """ Gets the PSD of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate:
        defaults to 44100 
      t_win: float
        length (in time) of each window
      t_shift: float
        length (in time) between the start of successive windows
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      wfft: any, Optional
        If you want to avoide recalculating the windowed fft, pass it in here!
  """
  # if you passed the wfft in then we'll skip over this
  if wfft is None:
    wfft = get_wfft(wf=wf, sample_rate=sample_rate, t_win=t_win, t_shift=t_shift, num_wins=num_wins)
    
  # we'll calculate the fft_freq manually
  num_win_pts = sample_rate * t_win
  sample_spacing = 1/sample_rate
  freq_ax = rfftfreq(num_win_pts, sample_spacing)

  
  # calculate necessary params from the wfft
  wfft_size = np.shape(wfft)
  num_win = wfft_size[0]
  num_freq_pts = wfft_size[1]

  # get phases
  phases = np.angle(wfft)

  # initialize array for phase diffs
  num_win_pairs = num_win - 1

  phase_diffs = np.zeros((num_win_pairs, num_freq_pts))
  
  for win in range(0, num_win_pairs):
    # take the difference between the phases in this current window and the next
    phase_diffs[win] = phases[win + 1] - phases[win]

  # get the sin and cos of the phase diffs, and average over the window pairs
  xx= np.mean(np.sin(phase_diffs),axis=0)
  yy= np.mean(np.cos(phase_diffs),axis=0)

  # finally, output the vector strength (for each frequency)
  coherence = np.sqrt(xx**2 + yy**2)

  if not return_all:
    return coherence
  else:
    return {  
      "freq_ax" : freq_ax,
      "coherence" : coherence
      }

def coherence_vs_psd(wf, sample_rate, t_win, t_shift=None, num_wins=None, max_vec_strength=1, psd_shift=0, db=True, xmin=None, xmax=None, 
                     ymin=None, ymax=None, wf_title=None, show_plot=True, do_coherence=True, do_psd=True, fig_num=1):
  """ Plots the power spectral density and phase coherence of an input waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
      t_win: float
        length (in time) of each window
      t_shift: float, Optional
        amount (in time) between the start points of adjacent windows. Defaults to t_win (aka no overlap)
      num_wins: int, Optional
        If this isn't passed, then it will just get the maximum number of windows of the given size
      max_vec_strength: int, Optional
        multiplier on the vector strength of phase coherence; defaults to 1
      psd_shift: int, Optional
        shifts the PSD up or down
      db: bool, Optional
        Chooses whether to plot PSD on a dB (10*log_10) scale
      xmin: float, Optional
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
  # get wfft so we don't have to do it twice below
  d = get_wfft(wf=wf, sample_rate=sample_rate, t_shift=t_shift, t_win=t_win, num_wins=num_wins, return_all=True)
  wfft = d["wfft"]
  freq_ax = d["freq_ax"]


  # get (averaged over windows) PSD
  psd = get_psd(wf, sample_rate, t_win, wfft=wfft)

  # get coherence
  coherence = get_coherence(wf, sample_rate, t_win, wfft=wfft)

  # PLOT!
  f = freq_ax
  
  if (db == True):
    psd = 20*np.log10(psd)
  psd = psd + psd_shift
  coherence = max_vec_strength*coherence

  # get 2 axes for double y axis
  ax1 = plt.subplots(num=fig_num)[1]
  ax2 = ax1.twinx()

  # plot + set labels
  if do_coherence:
    ax1.plot(f, coherence, label=f"Coherence: t_win={t_win}, t_shift={t_shift}", color='purple')
    ax1.set_xlabel('Freq [Hz]')
    ax1.set_ylabel('Phase Coherence', color='purple')
    ax1.legend(loc="lower left")
  if do_psd:
    ax2.plot(f, psd, label="PSD", color='r')
    ax2.set_xlabel('Freq [Hz]')
    ax2.set_ylabel('PSD [dB]', color='r')
    ax2.legend(loc="lower right")


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


def spectrogram(wf, sample_rate, t_win, t_shift=None, num_wins=None, db=True, cmap='rainbow', vmin=None, vmax=None,
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=True, fig_num=1):
  
  """ Plots a spectrogram of the waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
      t_win: float
        length (in time) of each window
      t_shift: float
        amount (in time) between the start points of adjacent windows. Defaults to t_win (aka no overlap)
      num_wins: int, Optional
        If this isn't passed, then it will just get the maximum number of windows of the given size
      db: bool, Optional
        Chooses whether to put PSD on a dB (10*log_10) scale
      cmap: str, Optional
        Sets cmap for the pcolormesh function
      vmin: float, Optional
        Sets min value of colorbar
      vmax: float, Optional
        Sets min value of colorbar
      xmin: float, Optional
        Defaults to 0
      xmax: float, Optional
      ymin: float, Optional
        Sets the PSD axis
      ymax: float, Optional
        Sets the PSD axis
      wf_title: String, Optional
        Plot title is: "Spectrogram of {wf_title}"
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional
  """
  # calculate the windowed fft, which outputs three arrays we will use
  wfft_output = get_wfft(wf, sample_rate=sample_rate, num_wins=num_wins, t_win=t_win, t_shift=t_shift, return_all=True)
  # this is the windowed fft itself
  wfft = wfft_output["wfft"]
  # this is the frequency axis
  freq_ax = wfft_output["freq_ax"]
  # these are the indices of where each window starts in the waveform 
  win_start_indices = wfft_output["win_start_indices"]
  # to convert these to time, just divide by sample rate 
  t_ax = win_start_indices / sample_rate
  # calculate the psd of each window
  win_psd = get_psd(wf, sample_rate=sample_rate, t_win=t_win, wfft=wfft, return_all=True)["win_psd"]
  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax) 

  # if db is passed in, convert psd to db
  if db:
      win_psd = 10*np.log10(win_psd)
  # plot!
  plt.figure(fig_num)
  # plot the colormesh (note we have to transpose win_psd since its first dimension - which picks the row of the matrix - is t. 
  # We want t on the x axis, meaning we want it to pick the column, not the row!  
  plt.pcolormesh(xx, yy, win_psd.T, vmin=vmin, vmax=vmax, cmap=cmap)
  label = "PSD"
  if db:
      label = label + " [dB]"
  plt.colorbar(label=label)
  plt.xlabel("Time")
  plt.ylabel("Frequency (Hz)")
  plt.xlim(xmin, xmax)
  plt.ylim(ymin, ymax)
  if wf_title:
      title = f"Spectrogram of {wf_title}: t_win={t_win}, t_shift={t_shift}"
  else: 
    title = f"Spectrogram: t_win={t_win}, t_shift={t_shift}"
  plt.title(title)
  if show_plot:
      plt.show()

def coherogram(wf, sample_rate, t_win, t_shift, scope=2, freq_ref_step=1, ref_type="next_win", num_wins=None, cmap='rainbow', vmin=None, vmax=None,
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=True, fig_num=1):
  
  """ Plots a coherogram of the waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
      t_win: float
        length (in time) of each window
      t_shift: float
        amount (in time) between the start points of adjacent windows. Defaults to t_win (aka no overlap)
      ref_type: str, Optional
        determines what to reference the phase of each window against:
        "next_win" for the same freq of the following window or "next_freq" for the next freq bin of the current window
      scope: int, Optional
        number of windows on either side to average over for vector strength
      freq_ref_step: int, Optional
        how many frequency bins over to use as a phase reference
      num_wins: int, Optional
        If this isn't passed, then it will just get the maximum number of windows of the given size
      cmap: str, Optional
        Sets cmap for the pcolormesh function
      vmin: float, Optional
        Sets min value of colorbar
      vmax: float, Optional
        Sets min value of colorbar
      xmin: float, Optional
        Defaults to 0
      xmax: float, Optional
      ymin: float, Optional
        Sets the PSD axis
      ymax: float, Optional
        Sets the PSD axis
      wf_title: String, Optional
        Plot title is: "Coherogram of {wf_title}"
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional
  """

  # calculate the windowed fft, which outputs three arrays we will use
  wfft_output = get_wfft(wf, sample_rate=sample_rate, num_wins=num_wins, t_win=t_win, t_shift=t_shift, return_all=True)
  # this is the windowed fft itself
  wfft = wfft_output["wfft"]
  # this is the frequency axis
  freq_ax = wfft_output["freq_ax"]
  # these are the indices of where each window starts in the waveform 
  win_start_indices = wfft_output["win_start_indices"]
  # get num_wins (if you passed in a num_wins, this will just redefine it at the same value!)
  num_wins = len(win_start_indices)
  # to convert these to time, just divide by sample rate 
  t_ax = win_start_indices / sample_rate

  if scope < 1:
    raise Exception("We need at least one window on either side to average over!")

  # restrict the time axis since we need "scope" # of windows on either side of t. 
    # note if scope = 1, then t_ax[1:-1] will keep the one at the 1 index but not the one at the last index 
    # this is because the : operator is [start, stop)... and this is what we want!
  t_ax = t_ax[scope:-scope]

  if ref_type == "next_win":
    # if using this reference method, since we must compare it to the subsequent window, 
    # we will not be able to calculate a coherence for the final t value
    t_ax = t_ax[0:-1] 
  elif ref_type == "next_freq":
    # if using this reference method, since we must compare each freq to the one freq_ref_step away, 
    # we will not be able to calculate a coherence for the final freq_ref_step # of values
    freq_ax = freq_ax[0:-freq_ref_step]
  
  # calc num_freqs
  num_freqs = len(freq_ax)
  # initialize matrix for coherences (taking the above considerations into account)
  coherences = np.zeros((len(t_ax), num_freqs))
  # get phase information from wfft
  phases = np.angle(wfft)
  
  if ref_type == "next_win":
    # initialize phase_diffs; the -1 is because we will not be able to calculate a phase diff for the final window!
    phase_diffs = np.zeros((num_wins - 1, num_freqs))
    for index in range(num_wins - 1):
      phase_diffs[index] = phases[index + 1] - phases[index]

  elif ref_type == "next_freq":
    # initialize phase_diffs:
      # no -1 for num_wins in contrast to above since we can use every window!)
    phase_diffs = np.zeros((num_wins, num_freqs))
    for win_index in range(num_wins):
      for freq_index in range(num_freqs):
        phase_diffs[win_index, freq_index] = phases[win_index, freq_index + freq_ref_step] - phases[win_index, freq_index]

  # now we calculate coherences (vector strengths) for each group of 2*scope + 1
  # If scope is 2, then we will start at index 2 (so we can grab the windows at 0 and 1 on the left - and right - sides. So scope of 2!)
  for k in range(scope, len(t_ax) + scope):
    # get the start and end indices of the group of size 2*scope + 1
    start = k - scope
    end = k + scope + 1
      # the + 1 is just because [start:end] doesn't include the endpoint!
    # take the sin/cos of the phase_diffs and average over the 0th axis (over the group of windows)
    xx = np.mean(np.sin(phase_diffs[start:end]), 0)
    yy = np.mean(np.cos(phase_diffs[start:end]), 0)
    # now when we input to coherences, we want to start at 0 and go to len(t_ax) so we use [k - scope] as our index
    coherences[k - scope] = np.sqrt(xx**2 + yy**2)
    # note that the first index of the t_ax will correspond to the first index of coherence since both were shifted in the same way!

  # Initialize plotting!
  plt.figure(fig_num)
  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax) 
  # plot the colormesh
    # note we have to transpose "coherences" since its first dimension - which picks the row of the matrix - is t and
    # we want t on the x axis, meaning we want it to pick the column, not the row!
      # of course, we could have defined the axes of coherences differently, but I think the way we have it now is much more intuitive.
      # pcolormesh is the silly one here. 
  plt.pcolormesh(xx, yy, coherences.T, vmin=vmin, vmax=vmax, cmap=cmap)
  # set limits
  plt.xlim(xmin, xmax)
  plt.ylim(ymin, ymax)
  # set titles and labels
  plt.xlabel("Time")
  plt.ylabel("Frequency (Hz)")
  plt.colorbar(label="Vector Strength")
  if wf_title:
      title = f"Coherogram of {wf_title}: ref_type={ref_type}, t_win={t_win}, t_shift={t_shift}, scope={scope}"
  else: 
    title = f"Coherogram: ref_type={ref_type}, t_win={t_win}, t_shift={t_shift}, scope={scope}"
  if ref_type == "next_freq":
    title = title + f", freq_ref_step={freq_ref_step}"
  plt.title(title)
  # show plot
  if show_plot:
      plt.show()
      
      
def phase_portrait(wf, wf_title="Sum of Oscillators"):
  xdot = np.imag(wf)
  x = np.real(wf)
  plt.plot(x, xdot)
  plt.title("Phase Portrait of " + wf_title)
  plt.grid()
  plt.show()


def coherence_vs_psd2(wf, sample_rate, t_win, t_shift=None, num_wins=None, max_vec_strength=1, psd_shift=0, db=True, xmin=None, xmax=None, 
                     ymin=None, ymax=None, wf_title=None, show_plot=True, do_coherence=True, do_psd=True, fig_num=1):
  """ Plots the power spectral density and phase coherence of an input waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sample_rate: int
      t_win: float
        length (in time) of each window
      t_shift: float, Optional
        amount (in time) between the start points of adjacent windows. Defaults to t_win (aka no overlap)
      num_wins: int, Optional
        If this isn't passed, then it will just get the maximum number of windows of the given size
      max_vec_strength: int, Optional
        multiplier on the vector strength of phase coherence; defaults to 1
      psd_shift: int, Optional
        shifts the PSD up or down
      db: bool, Optional
        Chooses whether to plot PSD on a dB (10*log_10) scale
      xmin: float, Optional
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
  # get wfft so we don't have to do it twice below
  d = get_wfft2(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins, return_all=True)
  wfft = d["wfft"]
  freq_ax = d["freq_ax"]


  # get (averaged over windows) PSD
  psd = get_psd(wf, sample_rate, t_win, wfft=wfft)

  # get coherence
  coherence = get_coherence(wf, sample_rate, t_win, wfft=wfft)

  # PLOT!
  f = freq_ax
  
  if (db == True):
    psd = 20*np.log10(psd)
  psd = psd + psd_shift
  coherence = max_vec_strength*coherence

  # get 2 axes for double y axis
  ax1 = plt.subplots(num=fig_num)[1]
  ax2 = ax1.twinx()

  # plot + set labels
  if do_coherence:
    ax1.plot(f, coherence, label=f"Coherence: t_win={t_win}, t_shift={t_shift}", color='purple')
    ax1.set_xlabel('Freq [Hz]')
    ax1.set_ylabel('Phase Coherence', color='purple')
    ax1.legend(loc="lower left")
  if do_psd:
    ax2.plot(f, psd, label="PSD", color='r')
    ax2.set_xlabel('Freq [Hz]')
    ax2.set_ylabel('PSD [dB]', color='r')
    ax2.legend(loc="lower right")


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
