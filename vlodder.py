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
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

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
      ss_sol = vod.sol[osc, vod.n_transient:]
      avg_position_amplitudes[osc] = np.sqrt(np.mean((ss_sol.real)**2))
      # get the average cluster frequency
      # first get the psd of this oscillator
      psd = get_psd_vod(vod, osc)
      # Now, the paper seems to indicate a proper average over each frequency's PSD:
      avg_cluster_freqs[osc] = np.average(vod.fft_freq, weights=psd)
      # But Beth's way was just to use the frequency which has the highest PSD peak
      #avg_cluster_freqs[osc] = vod.fft_freq[np.argmax(psd)]
    
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
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the win #)
  # defaults to osc = -1 which is the sum of oscillators
  if osc == -1:
    wf = vod.SOO_fft[:, :]
  else:
    wf = vod.every_fft[osc, :, :]
  
  # get phases
  phases = np.angle(wf)
  # initialize array for phase diffs
  phase_diffs = np.zeros((vod.num_wins - 1, vod.num_freq_points))
  
  for win in range(0, vod.num_wins - 1):
    # take the difference between the phases in this current win and the next
    phase_diffs[win] = phases[win + 1] - phases[win]

  # get the average sin and cos of the phase diffs
  xx= np.mean(np.sin(phase_diffs),axis=0)
  yy= np.mean(np.cos(phase_diffs),axis=0)

  # finally, output the vector strength (for each frequency)
  return np.sqrt(xx**2 + yy**2)

def get_psd_vod(vod: Vodscillator, osc=-1, window=-1):
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the win #)
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
  # first, we get our 2D array with all the FFTs - (the zeroth dimension of y is the win #)
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


def get_apc(vod=Vodscillator, cluster_width=0.005, f_min=0.0, f_max=10.0, f_resolution=0.001, num_wins=100, t_win=1, amp_weights=True):
  """ Calculates phase coherence using the instantaneous information from the analytic signal

  Parameters
  ------------
      cluster_width: float
        Defines how close an oscillator's avg freq can be to the target frequency to count in that freq's cluster
      f_min: float, Optional
      f_max: float, Optional
      f_resolution: float, Optional
        These define the min, max, and size of the frequency boxes
      t_win: float, Optional
        Size of the window (in t) to calculate phase coherence over (should be small or else all phases will drift over the window)
      amp_weights: bool, Optional
        For each frequency box, the average vector strength over all pairs is weighted by the pairs' instantaneous amplitude (averaged over the window)
      
  """
  # function to get all oscillators whose average oscillator is near a particular frequency over a given window
  def cluster():
    # take "derivatives" to get instantaenous frequencies
    inst_freqs = (np.diff(inst_phases) / (2.0*np.pi) * vod.sample_rate)
    clusters = np.zeros((num_wins, num_freqs, vod.num_osc))

    # pick a window
    for win in range(num_wins):
      # find average frequency for each oscillator
      avg_freqs = np.average(inst_freqs[:, win*n_win:(win+1)*n_win], axis=1)
      # for each frequency box we look through each oscillator to see which oscillators are close enough to that frequency
      for f in range(num_freqs):
        for osc in range(vod.num_osc):
          # # keep track of how many oscillators we've found in the cluster
          # osc_in_clusters = 0
          if abs(avg_freqs[osc] - freqs[f]) < cluster_width:
              # put the oscillator's index in the ith position
              clusters[win, f, osc] = 1
              # i += 1
    return clusters
  
  # get SS part of solution
  ss_sol = vod.sol[:, vod.n_transient:]
  #get analytic signals of each oscillator
  analytic_signals = hilbert(ss_sol.real, axis=1)
  # get phases and amps
  inst_phases = np.unwrap(np.angle(analytic_signals))
  inst_amps = np.abs(analytic_signals)
  # get # points in window
  n_win = int(t_win * vod.sample_rate)
  # generate frequency array
  freqs = np.arange(f_min, f_max, f_resolution) #apc stands for analytic phase coherence
  num_freqs = len(freqs)
  
  # get all clusters
  clusters = cluster()
  
  # initialize array to store all phase coherences
  all_phase_coherences = np.zeros((num_wins, num_freqs))

  for win in range(num_wins):
      for f in range(len(freqs)):
        print(f"Window {win}: Finding PC for {freqs[f]}Hz")
        # create list of osc_indices in the cluster
        osc_indices = np.where(clusters[win, f] == 1)[0]
        # if there's no oscillators in here, set the PC for this freq to 0 and break
        if len(osc_indices) <= 1:
          continue
        # generate all possible pairs of oscillators in our cluster
        pairs = list(combinations(osc_indices, 2))
        # init temp arrays to store vector strengths for each pair
        num_pairs = len(pairs)
        pairwise_vec_strengths = np.zeros(num_pairs)
        # if amp_weights is on, we also need to store the average amplitude for the pair over the window 
        if amp_weights:
          pairwise_amp_weights = np.zeros(num_pairs)
        
        # this variable will help us index the pairwise_ arrays:
        k = 0
        for pair in pairs:
            # get the inst phases for each oscillator over the window
            win_phases_osc1 = inst_phases[pair[0], win*n_win:(win+1)*n_win]
            win_phases_osc2 = inst_phases[pair[1], win*n_win:(win+1)*n_win]
            #calculate vector strength of the difference over the window
            xx = np.average(np.sin(win_phases_osc1 - win_phases_osc2))
            yy = np.average(np.cos(win_phases_osc1 - win_phases_osc2))
            # do final calculation and store away
            pairwise_vec_strengths[k] = np.sqrt(xx**2 + yy**2)
            # if we want to weight by amplitude
            if amp_weights:
              # get the inst amps for each oscillator over the window
              win_amps_osc1 = inst_amps[pair[0], win*n_win:(win+1)*n_win]
              win_amps_osc2 = inst_amps[pair[1], win*n_win:(win+1)*n_win]
              # average between the two oscillators and then average over time
              pairwise_amp_weights[k] = np.mean((win_amps_osc1 + win_amps_osc2)/2)
            # get ready for next loop
            k+=1
        # average over all pairs (possibly weighting by pairwise_amp_weights) and store away
        if amp_weights:
          # note that this way, the higher amp pairs contribute more to that cluster's average. 
          # BUT frequencies whose clusters who have exceptionally high amp are NOT weighted any higher than the others!
            # if they did, then we would just be artificially boosting PC for frequencies with lots of power!
          all_phase_coherences[win, f] = np.average(pairwise_vec_strengths, weights=pairwise_amp_weights)
        else:
          all_phase_coherences[win, f] = np.mean(pairwise_vec_strengths)

  # average over all t_wins and return
  return np.mean(all_phase_coherences, 0)