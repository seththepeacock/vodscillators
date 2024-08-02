import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.axes import Axes
from vodscillator import *
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

# define helper functions
def get_wfft(wf, sample_rate, t_win, t_shift=None, fcut=False, num_wins=None, return_all=False, hann=False):
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


  if hann:
    for k in range(num_wins):
      wfft[k, :] = rfft(windowed_wf[k, :]*np.hanning(n_win))
  else:
    for k in range(num_wins):
      wfft[k, :] = rfft(windowed_wf[k, :])
      
  if fcut:
      fcut_index = np.where(freq_ax > 200)[0][0]
      wfft2 = np.zeros((num_wins, num_freq_pts - fcut_index), dtype=complex)
      freq_ax2 = np.zeros(num_freq_pts - fcut_index + 1)
      for k in range(num_wins):
        wfft2[k,:] = wfft[k, fcut_index:]
        freq_ax2 = freq_ax[fcut_index:]
  
  else:
    wfft2 = wfft
    freq_ax2 = freq_ax
  
  if not return_all:
    return wfft2
  else: 
    return {  
      "wfft" : wfft2,
      "freq_ax" : freq_ax2,
      "win_start_indices" : win_start_indices
      }


def get_psd(wf, sample_rate, t_win, num_wins=None, wfft=None, freq_ax=None, return_all=False):
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
        If you want to avoid recalculating the windowed fft, pass it in here!
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the wfft (either pass in both or neither!)
      return_all: bool, Optional
        Defaults to only returning the PSD averaged over all windows; if this is enabled, then a dictionary is returned with keys:
        "avg_psd", "freq_ax", "win_psd"
  """
  # make sure we either have both or neither
  if (wfft is None and freq_ax is not None) or (wfft is not None and freq_ax is None):
    raise Exception("We need both wfft and freq_ax (or neither)!")
  
  # if you passed the wfft and freq_ax in then we'll skip over this
  if wfft is None:
    d = get_wfft(wf=wf, sample_rate=sample_rate, t_win=t_win, num_wins=num_wins, return_all=True)
    wfft = d["wfft"]
    freq_ax = d["freq_ax"]
  
  # calculate necessary params from the wfft
  wfft_size = np.shape(wfft)
  num_wins = wfft_size[0]
  num_freq_pts = wfft_size[1]

  # calculate the number of samples in the window for normalizing factor purposes
    # + 1 is because if you have SR=2 and you want a two second window, this will take 5 samples!
  n_win = int(t_win*sample_rate) + 1
  
  # initialize array
  win_psd = np.zeros((num_wins, num_freq_pts))

  # calculate the normalizing factor (canonical for discrete PSD)
  normalizing_factor = sample_rate * n_win
  # get PSD for each window
  for win in range(num_wins):
    win_psd[win, :] = ((np.abs(wfft[win, :]))**2) / normalizing_factor
  # average over all windows
  psd = np.mean(win_psd, 0)
  if not return_all:
    return psd
  else:
    return {  
      "psd" : psd,
      "freq_ax" : freq_ax,
      "win_psd" : win_psd
      }

def get_coherence(wf, sample_rate, fcut= False, t_win=16, t_shift=1, unwrap=False, bin_shift=1, num_wins=None, wfft=None, freq_ax=None, ref_type="next_win", return_all=False):
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
        length (in time) between the start of successive windows (primarily for next_)
      num_wins: int, Optional
        If this isn't passed, then just get the maximum number of windows of the given size
      wfft: any, Optional
        If you want to avoid recalculating the windowed fft, pass it in here!
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the wfft (either pass in both or neither!)
      ref_type: str, Optional
        Either "next_win" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      bin_shift: int, Optional
        How many bins over to reference phase against for next_freq
      return_all: bool, Optional
        Defaults to only returning the PSD averaged over all windows; if this is enabled, then a dictionary is returned with keys:
        "avg_psd", "freq_ax", "win_psd"
  """
  # define a helper function to calculate vector strength from an array of phase diffs (one for each freq)
  def get_vector_strength(phase_diffs):
    # get the sin and cos of the phase diffs, and average over the window pairs
    xx= np.mean(np.sin(phase_diffs),axis=0)
    yy= np.mean(np.cos(phase_diffs),axis=0)

    # finally, output the vector strength (for each frequency)
    return np.sqrt(xx**2 + yy**2)
  
  # make sure we either have both or neither
  if (wfft is None and freq_ax is not None) or (wfft is not None and freq_ax is None):
    raise Exception("We need both wfft and freq_ax (or neither)!")

  # default t_shift to t_win (no overlap)
  if t_shift is None:
    t_shift=t_win
  
  # if you passed the wfft and freq_ax in then we'll skip over this
  if wfft is None:
    d = get_wfft(wf=wf, sample_rate=sample_rate, fcut=fcut, t_win=t_win, t_shift=t_shift, num_wins=num_wins, return_all=True)
    wfft = d["wfft"]
    freq_ax = d["freq_ax"]
  
  # calculate necessary params from the wfft
  wfft_size = np.shape(wfft)
  num_wins = wfft_size[0]
  num_freq_pts = wfft_size[1]

  # get phases
  # phases = np.unwrap(np.angle(wfft))
  phases=np.angle(wfft)
  
  # we can reference each phase against the phase of the same frequency in the next window:
  if ref_type == "next_win":
    # initialize array for phase diffs; we won't be able to get it for the final window
    phase_diffs = np.zeros((num_wins - 1, num_freq_pts))
    
    # calc phase diffs
    for win in range(num_wins - 1):
      # take the difference between the phases in this current window and the next
      phase_diffs[win] = phases[win + 1] - phases[win]
    
    coherence = get_vector_strength(phase_diffs)
    
  # or we can reference it against the phase of the next frequency in the same window:
  elif ref_type == "next_freq":
    
    # Unwrap Phases (unless the bin_shift is not 1)
    if bin_shift==1:
      if unwrap:
        phases = np.unwrap(phases, axis=1)
      
    # initialize array for phase diffs; -bin_shift is because we won't be able to get it for the #(bin_shift) freqs
    phase_diffs = np.zeros((num_wins, num_freq_pts - bin_shift))
    # we'll also need to take the last #(bin_shift) bins off the freq_ax
    freq_ax = freq_ax[0:-bin_shift]
    
    # pass this in:
    special_unwrap_freq = None
    
    # optionally give special unwrapping treatment to a particular frequency bin 
      # so that the phase diffs between this and #(bin_shift) bins over are between -pi and pi
    if special_unwrap_freq is not None:
      special_freq_bin = int(special_unwrap_freq * t_win)
      k = int(np.mod(special_freq_bin, bin_shift))
      j = int(special_freq_bin / bin_shift)
      special_phases = np.unwrap(phases[:, k::bin_shift], 1)[:, j:j+2]  
      # calc phase diffs
      for win in range(num_wins):
        for freq_bin in range(num_freq_pts - bin_shift):
          phase_diffs[win, freq_bin] = phases[win, freq_bin + bin_shift] - phases[win, freq_bin]
          if freq_bin == special_freq_bin:
            phase_diffs[win, freq_bin] = special_phases[win, 1] - special_phases[win, 0]
    # or just do it the normal way!
    else:
      # calc phase diffs
      for win in range(num_wins):
        for freq_bin in range(num_freq_pts - bin_shift):
          phase_diffs[win, freq_bin] = phases[win, freq_bin + bin_shift] - phases[win, freq_bin]
    
    # get final coherence
    coherence = get_vector_strength(phase_diffs)
    
    #alter freq ax so that each coherence is the difference between half a bin on either side
    # freq_ax = freq_ax + (1/2)*(1/t_win)
  
  elif ref_type == "prev_freq":
    # unwrap it w.r.t. neighboring frequency bins
    phases=np.unwrap(phases, axis=1)
    # initialize array for phase diffs; - 1 is because we won't be able to get it for the final freq 
    phase_diffs = np.zeros((num_wins, num_freq_pts - 1))
    # we'll also need to take the first bin off the freq_ax
    freq_ax = freq_ax[1:]
    
    # calc phase diffs
    for win in range(num_wins):
      for freq_bin in range(1, num_freq_pts):
        # so the first entry is in phase_diffs[win, 0] and corresponds to the entry for phases[win, 1] which makes sense bc our first bin on freq_ax is the one that was originally at index 1
        phase_diffs[win, freq_bin - 1] = phases[win, freq_bin] - phases[win, freq_bin - 1]
    coherence = get_vector_strength(phase_diffs)
  
  # or we can reference it against the phase of both the lower and higher frequencies in the same window
  elif ref_type == "both_freqs":
    # initialize arrays
      # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
    pd_low = np.zeros((num_wins, num_freq_pts - 2))
    pd_high = np.zeros((num_wins, num_freq_pts - 2))
    # take the first and last bin off the freq ax
    freq_ax = freq_ax[1:-1]
    
    # calc phase diffs
    for win in range(num_wins):
      for freq_bin in range(1, num_freq_pts - 1):
        # the - 1 is so that we start our phase_diffs arrays at 0 and put in num_freq_pts-2 points. 
        # These will correspond to our new frequency axis.
        pd_low[win, freq_bin - 1] = phases[win, freq_bin] - phases[win, freq_bin - 1]
        pd_high[win, freq_bin - 1] = phases[win, freq_bin + 1] - phases[win, freq_bin]
    coherence_low = get_vector_strength(pd_low)
    coherence_high = get_vector_strength(pd_high)
    # average the coherences you would get from either of these
    coherence = (coherence_low + coherence_high)/2
    # set phase_diffs to one of these so it doesn't throw an error when you try to return phase_Diffs
    phase_diffs = pd_high
  else:
    raise Exception("You didn't input a valid ref_type!") 

  if not return_all:
    return coherence
  else:
    return {  
      "freq_ax" : freq_ax,
      "coherence" : coherence,
      "phase_diffs": phase_diffs,
      "phases": phases,
      "num_wins": num_wins
      }
      # define a helper function to be reused


def coherence_vs_psd(wf, sample_rate, t_win, t_shift=None, fcut=False, bin_shift=1, num_wins=None, khz=False, downsample_freq=False, ref_type="next_win", max_vec_strength=1, psd_shift=0, db=True, xmin=None, xmax=None, 
                     ymin=None, ymax=None, wf_title=None, show_plot=False, do_coherence=True, do_psd=True, do_means=False, ax=None, fig_num=1, hann=False):
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
      ref_type: str, Optional
        Either "next_win" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      bin_shift: int, Optional
        How many bins over to reference phase against for next_freq
      downsample_freq: int, Optional
        This will skip every "downsample_freq"-th frequency point (for comparing effect of t_win)
      max_vec_strength: int, Optional
        multiplier on the vector strength of phase coherence; defaults to 1
      psd_shift: int, Optional
        shifts the PSD up or down
      db: bool, Optional
        Chooses whether to plot PSD on a dB (10*log_10) scale
      xmin: float, Optional
      xmax: float, Optional
      ymin: float, Optional
        Sets the coherence y axis
      ymax: float, Optional
        Sets the coherence y axis
      wf_title: String, Optional
        Plot title is: "Phase Coherence and PSD of {wf_title}"
      wf_comp: str, Optional
      do_coherence: bool, Optional
        optionally suppress coherence plot
      do_psd: bool, Optional
        optionally suppress PSD plot
      do_means: bool, Optional
        Optionally plot <|phase diffs|>
      show_plot: bool, Optional
        Call plt.show() at the end
      ax: Axes, Optional
        Tells it to plot onto this Axes object 
      fig_num: int, Optional
        If you didn't pass in an Axes, then it will create a figure and this will set the figure number
  """
  # get default for t_shift
  if t_shift is None:
    t_shift = t_win
  # get wfft so we don't have to do it twice below
  d = get_wfft(wf=wf, sample_rate=sample_rate, t_shift=t_shift, fcut=fcut, t_win=t_win, num_wins=num_wins, return_all=True, hann=hann)
  wfft = d["wfft"]
  # we'll want to pass this through the subsequent functions as well to maintain correspondence through all the shifts
  freq_ax = d["freq_ax"]
  
  # get (averaged over windows) PSD
  p = get_psd(wf=wf, sample_rate=sample_rate, t_win=t_win, wfft=wfft, freq_ax=freq_ax, return_all=True)
  psd = p["psd"]
  psd_freq_ax = p["freq_ax"]

  # get coherence
  c = get_coherence(wf=wf, sample_rate=sample_rate, t_win=t_win, wfft=wfft, ref_type=ref_type, freq_ax=freq_ax, bin_shift=bin_shift, return_all=True)
  coherence = c["coherence"]
  coherence_freq_ax = c["freq_ax"]

  if downsample_freq:
    coherence=coherence[::downsample_freq]
    coherence_freq_ax=coherence_freq_ax[::downsample_freq]
    psd=psd[::downsample_freq]
    psd_freq_ax=psd_freq_ax[::downsample_freq]
  
  if khz:
    psd_freq_ax = psd_freq_ax / 1000
    coherence_freq_ax = coherence_freq_ax / 1000
  
  if (db == True):
    psd = 20*np.log10(psd)
  psd = psd + psd_shift
  coherence = max_vec_strength*coherence

  # if we haven't passed in an axes object, we'll initialize a figure and get the axes
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)
  
  # now we'll add an axes object with identical x-axis and empty y-axis (which we'll add the psd to)
  ax2 = ax.twinx()

  # plot + set labels
  if do_coherence:
    if ref_type == "next_freq":
      label = f"Phase Coherence: t_win={t_win}, t_shift={t_shift}, ref_type={ref_type}, bin_shift={bin_shift}"
    else:
      label = f"Phase Coherence: t_win={t_win}, t_shift={t_shift}, ref_type={ref_type}"
    # label="Phase Coherence"
    ax.plot(coherence_freq_ax, coherence, label=label, color='purple')
    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('Vector Strength', color='purple')
    ax.legend(loc="upper right")
  if do_means:
    phase_diffs = c["phase_diffs"]
    means = np.mean(np.abs(phase_diffs[:, :]), 0)
    ax.plot(coherence_freq_ax, means / np.pi, label="<|Phase Diffs|> / pi")
    ax.set_ylabel('Vector Strength and Normalized <|Phase Diffs|>', color = 'black')
    ax.legend(loc='upper right')
  
  if do_psd:
    ax2.plot(psd_freq_ax, psd, label="PSD", color='r')
    ax2.set_xlabel('Freq [Hz]')
    ax2.set_ylabel('PSD [dB]', color='r')
    ax2.legend(loc="lower right")
  
  if khz:
    ax.set_xlabel('Freq [kHz]')
    ax2.set_xlabel('Freq [kHz]')

  # set title
  if wf_title is None:
    wf_title = "Waveform"
  
  if do_means:
    ax.set_title(f"Phase Coherence, <|Phase Diffs|>, and PSD of {wf_title}")
  else:
    ax.set_title(f"Phase Coherence and PSD of {wf_title}")

  # finally, overwrite any default x and y lims (this does nothing if none were inputted)
  ax.set_xlim(left = xmin, right = xmax)
  ax.set_ylim(bottom = ymin, top = ymax)
  
  
  
  if show_plot:
    plt.show()
  
  return ax, ax2


def spectrogram(wf, sample_rate, t_win, t_shift=None, num_wins=None, db=True, khz=False, cmap='rainbow', vmin=None, vmax=None,
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=False, ax=None,fig_num=1):
  
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
      khz: bool, Optional
        Sets frequency axis to kHz
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
  win_psd = get_psd(wf, sample_rate=sample_rate, t_win=t_win, wfft=wfft, freq_ax=freq_ax, return_all=True)["win_psd"]
  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax)
  if khz:
    yy = yy / 1000 

  # if db is passed in, convert psd to db
  if db:
      win_psd = 10*np.log10(win_psd)
  # plot!
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)
  
  # plot the colormesh (note we have to transpose win_psd since its first dimension - which picks the row of the matrix - is t. 
  # We want t on the x axis, meaning we want it to pick the column, not the row!  
  heatmap = ax.pcolormesh(xx, yy, win_psd.T, vmin=vmin, vmax=vmax, cmap=cmap)
  # get and set label for cbar
  color_label = "PSD"
  if db:
      color_label = color_label + " [dB]"
  cbar = plt.colorbar(heatmap)
  cbar.set_label(color_label)
  
  # set axes labels and titles
  ax.set_xlabel("Time")
  if khz:
    ylabel = ("Frequency [kHz]")
  else:
    ylabel = ("Frequency [Hz]")
  ax.set_ylabel(ylabel)
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  if wf_title:
      title = f"Spectrogram of {wf_title}: t_win={t_win}, t_shift={t_shift}"
  else: 
    title = f"Spectrogram: t_win={t_win}, t_shift={t_shift}"
  ax.set_title(title)
  
  # optionally show plot!
  if show_plot:
      plt.show()

def coherogram(wf, sample_rate, t_win, t_shift, scope=2, freq_ref_step=1, ref_type="next_win", num_wins=None, khz=False, cmap='rainbow', vmin=None, vmax=None,
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=False, ax=None, fig_num=1):
  
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
      khz: bool, Optional
        Chooses to use kHz for the frequency axis
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
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)

  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax) 
  # optionally rescale freq ax
  if khz:
    yy = yy / 1000 
  
  # plot the heatmap
    # note we have to transpose "coherences" since its first dimension - which picks the row of the matrix - is t and
    # we want t on the x axis, meaning we want it to pick the column, not the row!
      # of course, we could have defined the axes of coherences differently, but I think the way we have it now is much more intuitive.
      # pcolormesh is the silly one here. 
  heatmap = ax.pcolormesh(xx, yy, coherences.T, vmin=vmin, vmax=vmax, cmap=cmap)

  # get and set label for cbar
  cbar = plt.colorbar(heatmap)
  cbar.set_label("Vector Strength")
  
  # set limits
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  
  # set axes labels and titles
  ax.set_xlabel("Time")
  if khz:
    ax.set_ylabel("Frequency [kHz]")
  else:
    ax.set_ylabel("Frequency [Hz]")
  if wf_title:
      title = f"Coherogram of {wf_title}: ref_type={ref_type}, t_win={t_win}, t_shift={t_shift}, scope={scope}"
  else: 
    title = f"Coherogram: ref_type={ref_type}, t_win={t_win}, t_shift={t_shift}, scope={scope}"
  if ref_type == "next_freq":
    title = title + f", freq_ref_step={freq_ref_step}"
  ax.set_title(title)
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
  
def scatter_phase_diffs(freq, wf, sample_rate, t_win, num_wins=None, unwrap=True, ref_type="next_freq", bin_shift=1, t_shift=None, wf_title="Waveform", ax=None):
    c = get_coherence(wf, ref_type=ref_type, sample_rate=sample_rate, unwrap=unwrap, num_wins=num_wins, t_win=t_win, t_shift=t_shift, bin_shift=bin_shift, return_all=True)
    num_wins = c["num_wins"]
    phase_diffs = c["phase_diffs"]
    # get the freq_bin_index - note this only works if we're using next_freq! Then the 0 index bin is 0 freq, 1 index -> 1/t_win, 2 index -> 2/t_win
        # so then if freq is 0.8 and t_win is 10, the 1 index is 1/10=0.1, the 2 index is 0.2, ... , the 8 index is 0.8. So int(freq*t_win) = int(8.0) = 8
        # if you don't know the exact freq bin, rounding up is good... if it looked like ~ 0.84 then int(freq*t_win) = int(8.4) = 8
    freq_bin_index = int(freq*t_win)
    
    if ax is None:
      plt.figure(1)
      ax = plt.gca()
    assert isinstance(ax, Axes)

      
    ax.scatter(range(num_wins), phase_diffs[:, freq_bin_index])
    ax.set_title(f"Next Freq Bin Phase Diffs for {wf_title} at {freq}Hz")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Phase Diff")

def scatter_phases(freq, wf, sample_rate, t_win, num_wins=None, ref_type="next_freq", bin_shift=1, t_shift=None, wf_title="Waveform", ax=None):
    c = get_coherence(wf, ref_type=ref_type, sample_rate=sample_rate, num_wins=num_wins, t_win=t_win, t_shift=t_shift, bin_shift=bin_shift, return_all=True)
    num_wins = c["num_wins"]
    phases = c["phases"]
    # get the freq_bin_index - note this only works if we're using next_freq! Then the 0 index bin is 0 freq, 1 index -> 1/t_win, 2 index -> 2/t_win
        # so then if freq is 0.8 and t_win is 10, the 1 index is 1/10=0.1, the 2 index is 0.2, ... , the 8 index is 0.8. So int(freq*t_win) = int(8.0) = 8
        # if you don't know the exact freq bin, rounding up is good... if it looked like ~ 0.84 then int(freq*t_win) = int(8.4) = 8
    freq_bin_index = int(freq*t_win)
    
    if ax is None:
      plt.figure(1)
      ax = plt.gca()
      print("hm")
    assert isinstance(ax, Axes)

    ax.scatter(range(num_wins), phases[:, freq_bin_index])
    ax.set_title(f"Phases for {wf_title} at {freq}Hz")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Phase")
  