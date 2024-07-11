import numpy as np
import matplotlib.pyplot as plt
from vodscillator import *

# Vlodder helper functions:
def vlodder(vod: Vodscillator, plot_type:str, osc=-1, window=-1, xmin=0, xmax=None, ymin=None, ymax=None, db=True, psd_shift=0, wf_comp="re", 
                   wf_ss=False, show_plot=True, fig_num=1, wf_title=None):
  """ Plots various plots from Vodscillator

  Parameters
  ------------
  vod: Vodscillator
    input vodscillator
  plot_type: String
    "coherence" plots phase coherence;
    "cluster" plots V&D style frequency clustering plots;
    "psd" plots power spectral density 
      (if window=-1, takes the average over all windows, otherwise just takes the psd of that window)
      (if osc=-1, adds up fft of each oscillator and then takes PSD);
    "pre_psd" takes PSD of each oscillator and THEN plots the sum of the PSDs;
    "coherence_vs_psd" plots phase coherence and PSD;
    "wf" plots a waveform
  osc: int, Optional
  xmin: float, Optional
    Defaults to 0
  xmax: float, Optional
  ymin: float, Optional
  ymax: float, Optional
  db: bool, Optional
    Choose whether PSD plots are on a dB scale
  psd_shift: any, Optional
    Shifts the whole PSD plot
  wf_comp: str, Optional
    Which component of waveform signal to plot: "re" or "im" for real or imaginary, respectively
  wf_ss: boolean, Optional
    If you only want the steady state part of the waveform solution
  show_plot: boolean , Optional
    Gives the ability to suppress the plt.show()
  fig_num: int, Optional
    Only required if plotting multiple figures
  wf_title: String, Optional
    
    """

  # get frequency and time axes
  f = vod.fft_freq
  t = vod.tpoints

  # initialize figure (with fig_num if you're making multiple plots)
  plt.figure(fig_num)

  if plot_type == "coherence_vs_psd":
    psd = get_psd_vod(vod=vod, osc=osc, window=window)
    coherence = get_coherence_vod(vod=vod, osc=osc)
    # get 2 axes for double y axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(f, coherence, label=f"Coherence", color='purple')
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
      y1 = 10*np.log10(get_psd_vod(vod, osc))
      y2 = get_coherence_vod(vod, osc)
      plt.plot(f, y1, color = "red", lw=1, label="Power")
      plt.plot(f, y2, color = "purple", lw=1, label='Phase Coherence')
      plt.xlabel('Frequency [Hz]')  
      plt.ylabel(f'Power [dB] / Vector Strength]')
      plt.legend() 

      # set title
      title = "Phase Coherence and PSD of "
      if osc == -1:
        title = title + "Summed Response"
      else:
        title = title (f"Oscillator #{osc}")
      if wf_title:
        title = title + f": {wf_title}"
      plt.title(title)

  if plot_type == "psd":
    y = get_psd_vod(vod=vod, osc=osc, window=window)
    if db:              
      y = 10*np.log10(y)
    # optionally artifically move psd_shift up or down
    y = y + psd_shift
    plt.plot(f, y, color = "red", lw=1)
    if db:
      plt.ylabel('PSD [dB]')
    else: 
      plt.ylabel('PSD')
    plt.xlabel('Frequency')
    # set title
    title = "Power Spectral Density of "
    if osc == -1:
      title = title + "Summed Response"
    else:
      title = title (f"Oscillator #{osc}")
    if wf_title:
      title = title + f": {wf_title}"
    plt.title(title)

  if plot_type == "amps":
    y = get_amps_vod(vod=vod, osc=osc, window=window)
    if db:              
      y = 20*np.log10(y)
    plt.plot(f, y, color = "red", lw=1)
    if db:
      plt.ylabel('Amplitude [dB]')
    else: 
      plt.ylabel('Amplitude')
    plt.xlabel('Frequency')
    # set title
    title = "Amplitude Spectrum of "
    if osc == -1:
      title = title + "Summed Response"
    else:
      title = title (f"Oscillator #{osc}")
    if wf_title:
      title = title + f": {wf_title}"
    plt.title(title)
  
  if plot_type == "pre_psd":
    y = 0
    for k in range(vod.num_osc):
      y += get_psd_vod(vod, k)
    if db:
      y = 10*np.log10(y)
    plt.plot(f, y, color = "red", lw=1)
    plt.ylabel('Density')
    plt.xlabel('Frequency')
    # set title
    plt.title("Summed Power Spectral Density of Each Oscillator")



  if plot_type == "coherence":
    y = get_coherence_vod(vod, osc)
    plt.plot(f, y, color = "purple", lw=1)
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

def get_psd_vod(vod: Vodscillator, osc=-1, window=-1):
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the interval #)
  if osc == -1:
    # if osc = -1 (the default) we want the summed (SOO) response!
    fft = vod.SOO_fft[:, :]
  else:
    fft = vod.every_fft[osc, :, :]

  # take the amplitude squared and normalize
  psd = ((np.abs(fft))**2) / (vod.n_win*vod.sample_rate)
  # hmmm this seems to be what V&D do:
  # psd = ((np.abs(fft))**2) / (vod.t_win)
  
  if window == -1:
    # average over windows
    psd = np.mean(psd, 0)
  else:
    psd = psd[window]
  return psd


def get_amps_vod(vod: Vodscillator, osc=-1, window=-1):
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the interval #)
  if osc == -1:
    # if osc = -1 (the default) we want the summed (SOO) response!
    wf = vod.SOO_fft[:, :]
  else:
    wf = vod.every_fft[osc, :, :]
  # take the amplitude squared and normalize
  amps = np.abs(wf)
  if window == -1:
    # average over windows
    amps = np.mean(amps, 0)
  else:
    # pick a average
    amps = amps[window]
  return amps
