import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.axes import Axes
from vodscillator import *
from scipy.fft import rfft, rfftfreq, fftshift
from scipy.signal.windows import get_window


# define helper functions

def get_avg_vector(phase_diffs):
  # get the sin and cos of the phase diffs, and average over the window pairs
  xx= np.mean(np.sin(phase_diffs),axis=0)
  yy= np.mean(np.cos(phase_diffs),axis=0)
  
  # finally, output the averaged vector's vector strength and angle with x axis (for each frequency) 
  return np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)
# I THINK THE ISSUE HERE IS THAT THE ANGLE OF THIS AVERAGED VECTOR IS NOT THE SAME AS THE AVERAGE OF THE ANGLES OF THE INDIVIDUAL VECTORS
def get_sfft(wf, sr, tau, xi=None, num_segs=None, fftshift_segs=False, win_type='boxcar'):
  """ Returns a dict with the segmented fft and associated freq ax of the given waveform

  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau: float
        length (in time) of each window
      xi: float
        length (in time) between the start of successive segments
      num_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      fftshift_segs: bool, Optional
        Shifts each time-domain window of the fft with fftshift()
      win_type: String, Optional
        Window to apply before the FFT
  """
  
  # if you didn't pass in xi we'll assume you want no overlap - each new window starts at the end of the last!
  if xi is None:
    xi=tau
  
  # calculate the number of samples in the window
  nperseg = int(tau*sr)

  # and the number of samples to shift
  n_shift = int(xi*sr)

  # get sample_spacing
  sample_spacing = 1/sr

  # first, get the last index of the waveform
  final_wf_index = len(wf) - 1
    # - 1 is because of zero-indexing!
  # next, we get what we would be the largest potential seg_start_index
  final_seg_start_index = final_wf_index - (nperseg-1)
    # start at the final_wf_index. we need to collect nperseg points. this final index is our first one, and then we need nperseg - 1 more. 
    # So we march back nperseg-1 points, and then THAT is the last_potential_seg_start_index!
  seg_start_indices = np.arange(0, final_seg_start_index + 1, n_shift)
    # the + 1 here is because np.arange won't ever include the "stop" argument in the output array... but it could include (stop - 1) which is just our final_seg_start_index!

  # if number of segments is passed in, we make sure it's less than the length of seg_start_indices
  if num_segs is not None:
    if num_segs > len(seg_start_indices):
      raise Exception("That's more segments than we can manage! Decrease num_segs!")
  else:
    # if no num_segs is passed in, we'll just use the max number of segments
    num_segs = len(seg_start_indices)

  segmented_wf = np.zeros((num_segs, nperseg))
  for k in range(num_segs):
    seg_start = seg_start_indices[k]
    seg_end = seg_start + nperseg
    # grab the waveform in this segment
    seg = wf[seg_start:seg_end]
    if fftshift_segs: # optionally swap the halves of the waveform to effectively center it in time
      seg = fftshift(seg)
    segmented_wf[k, :] = seg
    # note this grabs the wf at indices seg_start, seg_start+1, ..., seg_end-1
      # if there are 4 samples and tau=xi=1 and SR=1, then nperseg=2, n_shift=1 and
      # Thus the first window will be samples 0 and 1, the next 1 and 2...

  # Now we do the ffts!

  # get frequency axis 
  freq_ax = rfftfreq(nperseg, sample_spacing)
  num_freq_pts = len(freq_ax)
  
  # initialize segmented fft array
  sfft = np.zeros((num_segs, num_freq_pts), dtype=complex)
  
  # get ffts (applying a window first; boxcar is no window) 

  win = get_window(win_type, nperseg)
  for k in range(num_segs):
    sfft[k, :] = rfft(segmented_wf[k, :] * win)

    
  return {  
    "sfft" : sfft,
    "freq_ax" : freq_ax,
    "seg_start_indices" : seg_start_indices
    }


def get_spectrum(wf, sr, tau, num_segs=None, win_type='boxcar', scaling='density', sfft=None, freq_ax=None, return_all=True):
  """ Gets the spectrum of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau: float
        length (in time) of each window; used in get_sfft and to calculate normalizing factor
      num_segs: int, Optional
        Used in get_sfft;
          if this isn't passed, then just gets the maximum number of segments of the given size
      sfft: any, Optional
        If you want to avoid recalculating the segmented fft, pass it in here!
      win_type: String, Optional
        Window to apply before the FFT
      scaling: String, Optional
        "mags" (magnitudes) or "density" (PSD) or "spectrum" (power spectrum)
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the sfft (either pass in both or neither!)
      return_all: bool, Optional
        Defaults to only returning the spectrum averaged over all segments; if this is enabled, then a dictionary is returned with keys:
        "spectrum", "freq_ax", "win_spectrum"
  """
  # make sure we either have both or neither
  if (sfft is None and freq_ax is not None) or (sfft is not None and freq_ax is None):
    raise Exception("We need both sfft and freq_ax (or neither)!")
  
  # if you passed the sfft and freq_ax in then we'll skip over this
  if sfft is None:
    d = get_sfft(wf=wf, sr=sr, tau=tau, num_segs=num_segs, win_type=win_type)
    sfft = d["sfft"]
    freq_ax = d["freq_ax"]
  
  # calculate necessary params from the sfft
  sfft_size = np.shape(sfft)
  num_segs = sfft_size[0]
  num_freq_pts = sfft_size[1]

  # calculate the number of samples in the window for normalizing factor purposes
  nperseg = int(tau*sr)
  
  # initialize array
  win_spectrum = np.zeros((num_segs, num_freq_pts))
  
  # get spectrum for each window
  for win in range(num_segs):
    win_spectrum[win, :] = ((np.abs(sfft[win, :]))**2)
    
  # average over all segments (in power)
  spectrum = np.mean(win_spectrum, 0)
  
  window = get_window(win_type, nperseg)
  S1 = np.sum(window)
  S2 = np.sum(window**2)
  ENBW = nperseg * S2 / S1**2
  
  if scaling == 'mags':
    spectrum = np.sqrt(spectrum)
    normalizing_factor = 1 / S1
    
  elif scaling == 'spectrum':
    normalizing_factor = 1 / S1**2
    
  elif scaling == 'density':
    bin_width = 1/tau
    normalizing_factor = 1 / (S1**2 * ENBW * bin_width) # Start with spectrum scaling, then divide by bin width (times ENBW in samples)

  else:
    raise Exception("scaling must be 'mags', 'density', or 'spectrum'!")
  
  # Normalize; since this is an rfft, we should multiply by 2 
  spectrum = spectrum * 2 * normalizing_factor
  # Except DC bin should NOT be scaled by 2
  spectrum[0] = spectrum[0] / 2
  # Nyquist bin shouldn't either (this bin only exists if nperseg is even)
  if nperseg % 2 == 0:
    spectrum[-1] = spectrum[-1] / 2
  
  if not return_all:
    return spectrum
  else:
    return {  
      "spectrum" : spectrum,
      "freq_ax" : freq_ax,
      "win_spectrum" : win_spectrum
      }
    

def get_coherence(wf, sr, tau, xi=None, bin_shift=1, num_segs=None, win_type='boxcar', fftshift_segs=False, sfft=None, freq_ax=None, ref_type="next_win", return_all=False):
  """ Gets the phase coherence of the given waveform with the given window size

  Parameters
  ------------
      wf: array
        waveform input array
      sr:
        sample rate of waveform
      tau: float
        length (in time) of each window
      xi: float
        length (in time) between the start of successive segments (primarily for next_)
      num_segs: int, Optional
        If this isn't passed, then just get the maximum number of segments of the given size
      win_type: String, Optional
        Window to apply before the FFT
      fftshift_segs: bool, Optional
        Shifts each time-domain window of the fft with fftshift_segs
      sfft: any, Optional
        If you want to avoid recalculating the segmented fft, pass it in here!
      freq_ax: any, Optional
        We have to also pass in the freq_ax for the sfft (either pass in both or neither!)
      ref_type: str, Optional
        Either "next_win" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      bin_shift: int, Optional
        How many bins over to reference phase against for next_freq
      return_all: bool, Optional
        Defaults to only returning the coherence; if this is enabled, then a dictionary is returned with keys:
        d["coherence"] = coherence
        d["phases"] = phases
        d["phase_diffs"] = phase_diffs
        d["means"] = means
        d["avg_phase_diff"] = avg_phase_diff
        d["num_segs"] = num_segs
        d["freq_ax"] = freq_ax
        d["sfft"] = sfft
  """
  # define output dictionary to be returned (we'll add everything later)
  d = {}
  
  # make sure we either have both sfft and freq_ax or neither
  if (sfft is None and freq_ax is not None) or (sfft is not None and freq_ax is None):
    raise Exception("We need both sfft and freq_ax (or neither)!")
  
  # if you passed the sfft and freq_ax in then we'll skip over this
  if sfft is None:
    d = get_sfft(wf=wf, sr=sr, tau=tau, xi=xi, num_segs=num_segs, win_type=win_type, fftshift_segs=fftshift_segs)
    sfft = d["sfft"]
    freq_ax = d["freq_ax"]
    
  # default xi to tau (no overlap)
  if xi is None:
    xi=tau
  
  # calculate necessary params from the sfft
  sfft_size = np.shape(sfft)
  num_segs = sfft_size[0]
  num_freq_pts = sfft_size[1]

  # get phases
  phases=np.angle(sfft)
  
  # we can reference each phase against the phase of the same frequency in the next window:
  if ref_type == "next_win":
    # unwrap phases along time window axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=0)
    
    # initialize array for phase diffs; we won't be able to get it for the final window
    phase_diffs = np.zeros((num_segs - 1, num_freq_pts))
    
    # calc phase diffs
    for win in range(num_segs - 1):
      # take the difference between the phases in this current window and the next
      phase_diffs[win] = phases[win + 1] - phases[win]
    
    coherence, avg_phase_diff = get_avg_vector(phase_diffs)
    
  # or we can reference it against the phase of the next frequency in the same window:
  elif ref_type == "next_freq":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
      
    # initialize array for phase diffs; -bin_shift is because we won't be able to get it for the #(bin_shift) freqs
    phase_diffs = np.zeros((num_segs, num_freq_pts - bin_shift))
    # we'll also need to take the last #(bin_shift) bins off the freq_ax
    freq_ax = freq_ax[0:-bin_shift]
    
    # calc phase diffs
    for win in range(num_segs):
      for freq_bin in range(num_freq_pts - bin_shift):
        phase_diffs[win, freq_bin] = phases[win, freq_bin + bin_shift] - phases[win, freq_bin]
    
    # get final coherence
    coherence, avg_phase_diff = get_avg_vector(phase_diffs)
    
    # Since this references each frequency bin to its adjacent neighbor, we'll plot them w.r.t. the average frequency 
        # this corresponds to shifting everything over half a bin width (bin width is 1/tau)
    freq_ax = freq_ax + (1/2)*(1/tau)
    
  
  # or we can reference it against the phase of both the lower and higher frequencies in the same window
  elif ref_type == "both_freqs":
    # unwrap phases along the frequency bin axis (this won't affect coherence, but it's necessary for <|phase diffs|>)
    phases = np.unwrap(phases, axis=1)
    
    # initialize arrays
      # even though we only lose ONE freq point with lower and one with higher, we want to get all the points we can get from BOTH so we do - 2
    pd_low = np.zeros((num_segs, num_freq_pts - 2))
    pd_high = np.zeros((num_segs, num_freq_pts - 2))
    # take the first and last bin off the freq ax
    freq_ax = freq_ax[1:-1]
    
    # calc phase diffs
    for win in range(num_segs):
      for freq_bin in range(1, num_freq_pts - 1):
        # the - 1 is so that we start our phase_diffs arrays at 0 and put in num_freq_pts-2 points. 
        # These will correspond to our new frequency axis.
        pd_low[win, freq_bin - 1] = phases[win, freq_bin] - phases[win, freq_bin - 1]
        pd_high[win, freq_bin - 1] = phases[win, freq_bin + 1] - phases[win, freq_bin]
    coherence_low, _ = get_avg_vector(pd_low)
    coherence_high, _ = get_avg_vector(pd_high)
    # average the coherences you would get from either of these
    coherence = (coherence_low + coherence_high)/2
    # set the phase diffs to one of these (could've also been pd_high)
    phase_diffs = pd_low
    
  else:
    raise Exception("You didn't input a valid ref_type!")
  
  
  
  # get <|phase diffs|>
    # note we're unwrapping w.r.t. the frequency axis
  means = np.mean(np.abs(phase_diffs), 0)
  
  if not return_all:
    return coherence
  
  else:
    d["coherence"] = coherence
    d["phases"] = phases
    d["phase_diffs"] = phase_diffs
    d["means"] = means
    d["avg_phase_diff"] = avg_phase_diff
    d["num_segs"] = num_segs
    d["freq_ax"] = freq_ax
    d["sfft"] = sfft
    return d
    


def coherence_vs_spectrum(wf, sr, tau, xi=None, bin_shift=1, num_segs=None, scaling='density', ref_type="next_win", win_type='boxcar', fftshift_segs=False, khz=False, db=True, downsample_freq=False, 
                     xmin=None, xmax=None, ymin=None, ymax=None, wf_title=None, slabel=False, do_coherence=True, do_spectrum=True, do_means=False, ax=None, fig_num=1):
  """ Plots the power spectral density and phase coherence of an input waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau: float
        length (in time) of each window
      xi: float, Optional
        amount (in time) between the start points of adjacent segments. Defaults to tau (aka no overlap)
      bin_shift: int, Optional
        How many bins over to reference phase against for next_freq, defaults to 1
      num_segs: int, Optional
        If this isn't passed, then it will just get the maximum number of segments of the given size
      ref_type: str, Optional
        Either "next_win" to ref phase against next window or "next_freq" for next frequency bin or "both_freqs" to compare freq bin on either side
      hann: bool, Optional
        Applies a hanning window before FFT
      downsample_freq: int, Optional
        This will skip every "downsample_freq"-th frequency point (for comparing effect of tau)
      khz: bool, Optional
        Plots frequency in kHz
      db: bool, Optional
        Plots spectrum on a dB (10*log_10) scale
      xmin: float, Optional
      xmax: float, Optional
      ymin: float, Optional
        Sets the coherence y axis
      ymax: float, Optional
        Sets the coherence y axis
      wf_title: String, Optional
        Plot title is: "Phase Coherence and spectrum of {wf_title}"
      slabel: bool, Optional
        Makes the label just "Phase Coherence" for cleaner plots
      do_coherence: bool, Optional
        Optionally suppress coherence plot
      do_spectrum: bool, Optional
        Optionally suppress spectrum plot
      do_means: bool, Optional
        Optionally plot <|phase diffs|>
      ax: Axes, Optional
        Tells it to plot onto this Axes object 
      fig_num: int, Optional
        If you didn't pass in an Axes, then it will create a figure and this will set the figure number
  """
  # get default for xi
  if xi is None:
    xi = tau
  # get sfft so we don't have to do it twice below
  d = get_sfft(wf=wf, sr=sr, tau=tau, xi=xi, num_segs=num_segs, win_type=win_type)
  sfft = d["sfft"]
  # we'll want to pass this through the subsequent functions as well to maintain correspondence through all the shifts
  freq_ax = d["freq_ax"]
  
  # get (averaged over segments) spectrum
  p = get_spectrum(wf=wf, sr=sr, tau=tau, sfft=sfft, scaling=scaling, win_type=win_type, fftshift_segs=fftshift_segs, freq_ax=freq_ax, return_all=True)
  spectrum = p["spectrum"]
  spectrum_freq_ax = p["freq_ax"]

  # get coherence
  c = get_coherence(wf=wf, sr=sr, tau=tau, sfft=sfft, ref_type=ref_type, freq_ax=freq_ax, win_type=win_type, fftshift_segs=fftshift_segs, bin_shift=bin_shift, return_all=True)
  coherence = c["coherence"]
  coherence_freq_ax = c["freq_ax"]

  if downsample_freq:
    coherence=coherence[::downsample_freq]
    coherence_freq_ax=coherence_freq_ax[::downsample_freq]
    spectrum=spectrum[::downsample_freq]
    spectrum_freq_ax=spectrum_freq_ax[::downsample_freq]
    coherence_freq_ax=coherence_freq_ax[::downsample_freq]
  
  if khz:
    spectrum_freq_ax = spectrum_freq_ax / 1000
    coherence_freq_ax = coherence_freq_ax / 1000
    xlabel="Frequency [kHz]"
  else:
    xlabel="Frequency[Hz]"
  
  if db:
    spectrum = 20*np.log10(spectrum)

  # if we haven't passed in an axes object, we'll initialize a figure and get the axes
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)
  
  # now we'll add an axes object with identical x-axis and empty y-axis (which we'll add the spectrum to)
  ax2 = ax.twinx()

  # this will collect the plots for legend labels
  p = []

  # plot + set labels
  if do_coherence:
    if ref_type == "next_freq":
      label = f"Phase Coherence: tau={tau}, xi={xi}, ref_type={ref_type}, bin_shift={bin_shift}"
    else:
      label = f"Phase Coherence: tau={tau}, xi={xi}, ref_type={ref_type}"
    if slabel:
      label = "Phase Coherence"
    p1 = ax.plot(coherence_freq_ax, coherence, label=label, color='purple')
    p = p + p1
    ax.set_ylabel('Vector Strength', color='purple')
    ax.set_ylim(0, 1)

  if do_means:
    phase_diffs = c["phase_diffs"]
    means = np.mean(np.abs(phase_diffs)/np.pi, 0)
    label = r"$\langle|\phi_j^{{\theta}}|\rangle/\pi$"
    p2 = ax.plot(coherence_freq_ax, means, label=label, color='C0')
    p = p + p2
    if not do_coherence:
      ax.set_ylabel(label, color='C0')
    else:
      ax.set_ylabel("Vector Strength, " + label, color='black')
    
  if do_spectrum:
    p3 = ax2.plot(spectrum_freq_ax, spectrum, label="spectrum", color='r')
    p = p + p3
    ax2.set_ylabel('spectrum [dB]', color='r')

  # add legends and titles
  labs = [l.get_label() for l in p]
  ax.legend(p, labs, loc="lower right", fontsize="8")
  
  ax.set_xlabel(xlabel)
  ax2.set_xlabel(xlabel)
  if wf_title is None:
    wf_title = "Waveform"
  
  if do_means:
    ax.set_title(f"Phase Coherence, $\langle|\phi_j^{{\theta}}|\rangle$, and spectrum of {wf_title}")
  else:
    ax.set_title(f"Phase Coherence and spectrum of {wf_title}")

  # finally, overwrite any default x and y lims (this does nothing if none were inputted)
  ax.set_xlim(left = xmin, right = xmax)
  ax2.set_ylim(bottom = ymin, top = ymax)
  # now we alter the coherence y axis
  ax.set_ylim(bottom = 0, top = 1.2)
  

  
  return ax, ax2


def spectrogram(wf, sr, tau, xi=None, num_segs=None, db=True, fftshift_segs=False, khz=False, cmap='rainbow', vmin=None, vmax=None, scaling='density',
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=False, ax=None,fig_num=1):
  
  """ Plots a spectrogram of the waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
        sample rate of waveform
      tau: float
        length (in time) of each window
      xi: float
        amount (in time) between the start points of adjacent segments. Defaults to tau (aka no overlap)
      num_segs: int, Optional
        If this isn't passed, then it will just get the maximum number of segments of the given size
      db: bool, Optional
        Chooses whether to put spectrum on a dB (10*log_10) scale
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
        Sets the spectrum axis
      ymax: float, Optional
        Sets the spectrum axis
      wf_title: String, Optional
        Plot title is: "Spectrogram of {wf_title}"
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional
  """
  # calculate the segmented fft, which outputs three arrays we will use
  sfft_output = get_sfft(wf, sr=sr, num_segs=num_segs, fftshift_segs=fftshift_segs, tau=tau, xi=xi)
  # this is the segmented fft itself
  sfft = sfft_output["sfft"]
  # this is the frequency axis
  freq_ax = sfft_output["freq_ax"]
  # these are the indices of where each window starts in the waveform 
  seg_start_indices = sfft_output["seg_start_indices"]
  # to convert these to time, just divide by sample rate 
  t_ax = seg_start_indices / sr
  # calculate the spectrum of each window
  win_spectrum = get_spectrum(wf, sr=sr, tau=tau, sfft=sfft, freq_ax=freq_ax, fftshift_segs=fftshift_segs, scaling=scaling, return_all=True)["win_spectrum"]
  # make meshgrid
  xx, yy = np.meshgrid(t_ax, freq_ax)
  if khz:
    yy = yy / 1000 

  # if db is passed in, convert spectrum to db
  if db:
      win_spectrum = 10*np.log10(win_spectrum)
  # plot!
  if ax is None:
    plt.figure(fig_num)
    ax = plt.gca()
  assert isinstance(ax, Axes)
  
  # plot the colormesh (note we have to transpose win_spectrum since its first dimension - which picks the row of the matrix - is t. 
  # We want t on the x axis, meaning we want it to pick the column, not the row!  
  heatmap = ax.pcolormesh(xx, yy, win_spectrum.T, vmin=vmin, vmax=vmax, cmap=cmap)
  # get and set label for cbar
  color_label = "spectrum"
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
      title = f"Spectrogram of {wf_title}: tau={tau}, xi={xi}"
  else: 
    title = f"Spectrogram: tau={tau}, xi={xi}"
  ax.set_title(title)
  
  # optionally show plot!
  if show_plot:
      plt.show()

def coherogram(wf, sr, tau, xi, scope=2, freq_ref_step=1, ref_type="next_win", num_segs=None, fftshift_segs=False, khz=False, cmap='rainbow', vmin=None, vmax=None,
                xmin=0, xmax=None, ymin=None, ymax=None, wf_title=None, show_plot=False, ax=None, fig_num=1):
  
  """ Plots a coherogram of the waveform
  
  Parameters
  ------------
      wf: array
        waveform input array
      sr: int
      tau: float
        length (in time) of each window
      xi: float
        amount (in time) between the start points of adjacent segments. Defaults to tau (aka no overlap)
      ref_type: str, Optional
        determines what to reference the phase of each window against:
        "next_win" for the same freq of the following window or "next_freq" for the next freq bin of the current window
      scope: int, Optional
        number of segments on either side to average over for vector strength
      freq_ref_step: int, Optional
        how many frequency bins over to use as a phase reference
      num_segs: int, Optional
        If this isn't passed, then it will just get the maximum number of segments of the given size
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
        Sets the spectrum axis
      ymax: float, Optional
        Sets the spectrum axis
      wf_title: String, Optional
        Plot title is: "Coherogram of {wf_title}"
      show_plot: bool, Optional
        Repress showing plot
      fig_num: int, Optional
  """

  # calculate the segmented fft, which outputs three arrays we will use
  sfft_output = get_sfft(wf, sr=sr, num_segs=num_segs, tau=tau, xi=xi, fftshift_segs=fftshift_segs)
  # this is the segmented fft itself
  sfft = sfft_output["sfft"]
  # this is the frequency axis
  freq_ax = sfft_output["freq_ax"]
  # these are the indices of where each window starts in the waveform 
  seg_start_indices = sfft_output["seg_start_indices"]
  # get num_segs (if you passed in a num_segs, this will just redefine it at the same value!)
  num_segs = len(seg_start_indices)
  # to convert these to time, just divide by sample rate 
  t_ax = seg_start_indices / sr

  if scope < 1:
    raise Exception("We need at least one window on either side to average over!")

  # restrict the time axis since we need "scope" # of segments on either side of t. 
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
  # get phase information from sfft
  phases = np.angle(sfft)
  
  if ref_type == "next_win":
    # initialize phase_diffs; the -1 is because we will not be able to calculate a phase diff for the final window!
    phase_diffs = np.zeros((num_segs - 1, num_freqs))
    for index in range(num_segs - 1):
      phase_diffs[index] = phases[index + 1] - phases[index]

  elif ref_type == "next_freq":
    # initialize phase_diffs:
      # no -1 for num_segs in contrast to above since we can use every window!)
    phase_diffs = np.zeros((num_segs, num_freqs))
    for win_index in range(num_segs):
      for freq_index in range(num_freqs):
        phase_diffs[win_index, freq_index] = phases[win_index, freq_index + freq_ref_step] - phases[win_index, freq_index]

  # now we calculate coherences (vector strengths) for each group of 2*scope + 1
  # If scope is 2, then we will start at index 2 (so we can grab the segments at 0 and 1 on the left - and right - sides. So scope of 2!)
  for k in range(scope, len(t_ax) + scope):
    # get the start and end indices of the group of size 2*scope + 1
    start = k - scope
    end = k + scope + 1
      # the + 1 is just because [start:end] doesn't include the endpoint!
    # take the sin/cos of the phase_diffs and average over the 0th axis (over the group of segments)
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
      title = f"Coherogram of {wf_title}: ref_type={ref_type}, tau={tau}, xi={xi}, scope={scope}"
  else: 
    title = f"Coherogram: ref_type={ref_type}, tau={tau}, xi={xi}, scope={scope}"
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
  
def scatter_phase_diffs(freq, wf, sr, tau, num_segs=None, ref_type="next_freq", win_type='boxcar', bin_shift=1, xi=None, fftshift_segs=False, wf_title="Waveform", ax=None):
    c = get_coherence(wf, ref_type=ref_type, sr=sr, win_type=win_type, num_segs=num_segs, tau=tau, xi=xi, bin_shift=bin_shift, fftshift_segs=fftshift_segs, return_all=True)
    num_segs = c["num_segs"]
    phase_diffs = c["phase_diffs"]
    # get the freq_bin_index - note this only works if we're using next_freq! Then the 0 index bin is 0 freq, 1 index -> 1/tau, 2 index -> 2/tau
        # so then if freq is 0.8 and tau is 10, the 1 index is 1/10=0.1, the 2 index is 0.2, ... , the 8 index is 0.8. So int(freq*tau) = int(8.0) = 8
        # if you don't know the exact freq bin, rounding up is good... if it looked like ~ 0.84 then int(freq*tau) = int(8.4) = 8
    freq_bin_index = int(freq*tau)
    
    if ax is None:
      plt.figure(1)
      ax = plt.gca()
    assert isinstance(ax, Axes)

      
    ax.scatter(range(num_segs), phase_diffs[:, freq_bin_index])
    ax.set_title(f"Next Freq Bin Phase Diffs (at {freq}Hz Bin) for {wf_title}")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Phase Diff")
    
    print("<|phase diffs|> = " + str(np.mean(np.abs(phase_diffs))))

def scatter_phases(freq, wf, sr, tau, num_segs=None, ref_type="next_freq", bin_shift=1, xi=None, fftshift_segs=False, wf_title="Waveform", ax=None):
    c = get_coherence(wf, ref_type=ref_type, sr=sr, num_segs=num_segs, tau=tau, xi=xi, bin_shift=bin_shift, fftshift_segs=fftshift_segs, return_all=True)
    num_segs = c["num_segs"]
    phases = c["phases"]
    # get the freq_bin_index - note this only works if we're using next_freq! Then the 0 index bin is 0 freq, 1 index -> 1/tau, 2 index -> 2/tau
        # so then if freq is 0.8 and tau is 10, the 1 index is 1/10=0.1, the 2 index is 0.2, ... , the 8 index is 0.8. So int(freq*tau) = int(8.0) = 8
        # if you don't know the exact freq bin, rounding up is good... if it looked like ~ 0.84 then int(freq*tau) = int(8.4) = 8
    freq_bin_index = int(freq*tau)
    
    if ax is None:
      plt.figure(1)
      ax = plt.gca()
      print("hm")
    assert isinstance(ax, Axes)

    ax.scatter(range(num_segs), phases[:, freq_bin_index])
    ax.set_title(f"Phases for {wf_title} at {freq}Hz")
    ax.set_xlabel("Window #")
    ax.set_ylabel("Phase")
  