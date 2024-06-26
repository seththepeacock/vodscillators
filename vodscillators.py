import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq

class Vodscillator:
  """
  Vod-structions
  1. Initialize
  2. Create Frequency Distribution with "set_freq"
  3. Set ICs with "set_ICs" 
    - requires #2 first!
  4. Generate Noise with "gen_noise"
  5. Set parameters in ODE with "set_ODE"
  6. Solve ODE Function with "solve_ODE" 
  7. Get summed solution with "sum_solution"
  8. Save your vodscillator to a file
  9. Plot
  """

  #Note - you must create a frequency distribution before setting ICs since an oscillator's IC depends on its char freq

  def __init__(s, **p):
    # s = self (refers to the object itself)
    # **p unpacks the dictionary of parameters (p) we pass into the initializer

    # GENERAL PARAMETERS
    s.num_osc = p["num_osc"]  # number of oscillators in chain [default = 100 or 150]
    
    if "name" in p:
      s.name = p["name"] # name your vodscillator!

  def set_freq(s, **p):
    # Freq Dist - creates s.omegas[]

    # NECESSARY PARAMETERS
    s.freq_dist = p["freq_dist"] # set to "linear" for frequency increasing linearly, set to "exp" for frequency increasing exponentially
    s.roughness_amp = p["roughness_amp"] # variation in each oscillator's characteristic frequency
    s.omega_0 = p["omega_0"]  # char frequency of lowest oscillator [default = 2*np.pi] 
    s.omega_N = p["omega_N"]  # char frequency of highest oscillator [default = 5*(2*np.pi)] 

    # Now we set the frequencies of each oscillator in our chain - linear or exponential
    if s.freq_dist == "linear":
      s.omegas = np.linspace(s.omega_0,s.omega_N,s.num_osc) # linearly spaced frequencies from omega_0 to omega_N
      # add roughness
    elif s.freq_dist == "exp":
      s.omegas = np.zeros(s.num_osc, dtype=float)
      for k in range(s.num_osc): # exponentially spaced frequencies from omega_0 to omega_N
        A = s.omega_0
        B = (s.omega_N/s.omega_0)**(1/(s.num_osc - 1))
        s.omegas[k] = A*(B**k)
        # note that omegas[0] = A = omega_0, and omegas[num_osc - 1] = A * B = omega_N

    # add roughness
    r = s.roughness_amp
    roughness = np.random.uniform(low=-r, high=r, size=(s.num_osc,))
    s.omegas = s.omegas + roughness

  def set_ICs(s, **p):
    # Generates s.ICs[]
    # Note: we need omegas[] for this, so set_freq first!  

    # NECESSARY PARAMETER
    s.IC_method = p["IC_method"]  # set to "rand" for randomized initial conditions, set to "const" for constant initial conditions
    
    s.ICs = np.zeros(s.num_osc, dtype=complex)

    if s.IC_method == "rand":
      # generate random ICs
      for k in range(s.num_osc):
        x_k = np.random.uniform(-1, 1)
        y_k = np.random.uniform(0, 1) # this was (0,1) in Beth's Code, not sure why yet
        s.ICs[k] = complex(x_k, y_k) # make a complex combination of x and y and save it in ICs
    elif s.IC_method == "const":
      # generate predetermined ICs
      all_x = np.linspace(-1, 1, s.num_osc)
      all_y = np.linspace(1, -1, s.num_osc)
      for k in range(s.num_osc):
        x_k = all_x[k]
        y_k = all_y[k]
        s.ICs[k] = x_k - (y_k*1j/s.omegas[k])

  def gen_noise(s, **p):
    # Generating Noise - creates s.xi_glob and s.xi_loc[]

    # NECESSARY PARAMETERS
    s.loc_noise_amp = p["loc_noise_amp"] # amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
    s.glob_noise_amp = p["glob_noise_amp"] # amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
    s.ti = p["ti"] # start time; [default = 0]
    s.n_transient = p["n_transient"]  # the # of time points we give for transient behavior to settle down; around 30000 [default = 35855]
    s.n_ss = p["n_ss"]  # the # of time points in a given interval of ss observation [default = 8192]
    s.num_intervals = p["num_intervals"] # [default for no noise is 1; with noise we must average over multiple intervals]
    s.sample_rate = p["sample_rate"] # [default = 128]

    # Calculate other params
    s.h = 1/s.sample_rate #delta t between time points
    s.tf = s.h*(s.n_transient + s.num_intervals * s.n_ss)  # end time is delta t * total # points

    # We want a global xi(t) and then one for each oscillator. 


    # First, generate time points
    s.tpoints = np.arange(s.ti, s.tf, s.h)

    # global --> will impact each oscillator equally at each point in time (e.g., wind blowing?)
    # first we randomly generate points uniformly within the given amplitude range
    global_noise = np.random.uniform(-s.glob_noise_amp, s.glob_noise_amp, len(s.tpoints)) 
    # then interpolate between (using a cubic spline) for ODE solving adaptive step purposes
    s.xi_glob = CubicSpline(s.tpoints, global_noise)

    # local --> will impact each oscillator differently at each point in time (e.g., brownian motion of fluid in inner ear surrounding hair cells)
    s.xi_loc = np.empty(s.num_osc, dtype=CubicSpline)
    for k in range(s.num_osc):
      # again, we randomly generate points uniformly within the given (local) amplitude range, then interpolate between
      local_noise = np.random.uniform(-s.loc_noise_amp, s.loc_noise_amp, len(s.tpoints))
      s.xi_loc[k] = CubicSpline(s.tpoints, local_noise)


  def solve_ODE(s, **p):
    # Setting parameters and integrating ODE system

    # NECESSARY PARAMETERS
    s.epsilon = p["epsilon"] # [default = 1.0] --> control parameter
    s.d_R = p["d_R"]  # [default = 0.15] --> real part of coupling coefficient
    s.d_I = p["d_I"]  # [default = -1.0] --> imaginary part of coupling coefficient
    s.B = p["B"] # [default = 1.0] --> amount of cubic nonlinearity

    # Numerically integrate our ODE from ti to tf with sample rate 1/h
    s.tpoints = np.arange(s.ti, s.tf, s.h) # array of time points
    s.sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y 
    # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
    # so s.sol[2][1104] is the value of the solution for the 3rd oscillator at the 1105th time point.
  
  def sum_solution(s):
    # this gives us the summed response of all the oscillators
    s.summed_sol = np.zeros(len(s.tpoints))
    for k in range(s.num_osc):
      s.summed_sol = s.summed_sol + s.sol[k]

  def get_fft(s):
    """ Returns four arrays:
    1. every_fft[oscillator index][ss interval index][output]
    2. summed_fft[output]
    3. AOI_fft_amps[oscillator index][output]
    4. summed_AOI_fft_amps[output]

    AOI = Averaged Over Intervals (for noise)

    """
    # first, we want to restrict our solution to after the system has entered steady state (ss).
    # we generate an array which is like our solution array, except with only timepoints after n_transient
    s.ss_sol = s.sol[:, s.n_transient:]

    # get frequency axis: the # of frequencies the fft checks depends on the # signal points we give it (n_ss), 
    # and sample spacing (h) tells it what these frequencies correspond to in terms of real time 
    s.fft_freq = rfftfreq(s.n_ss, s.h)
    s.num_freq_points = len(s.fft_freq)
    
    # compute the (r)fft for all oscillators individually and store them in "every_fft"
      # note we are taking the r(eal)fft since (presumably) we don't lose much information by only considering the real part (position) of the oscillators  
    s.every_fft = np.zeros((s.num_osc, s.num_intervals, s.num_freq_points), dtype=complex) # every_fft[which oscillator][which ss interval][output of fft]

    # we'll also add them all together to get the fft of the summed response (sum of fft's = fft of sum)
    s.summed_fft = np.zeros((s.num_intervals, s.num_freq_points), dtype=complex)

    # along the way, for each oscillator we'll calculate fft amplitudes and average over the ss intervals and store it in this array:
    s.AOI_fft_amps = np.zeros((s.num_osc, s.num_freq_points))

    # we'll also get the fft amplitude of the summed response averaged over all ss intervals
    s.summed_AOI_fft_amps = np.zeros(s.num_freq_points)

    for interval in range(s.num_intervals):
      for osc in range(s.num_osc):
        # calculate fft
        s.every_fft[osc][interval] = rfft((s.ss_sol[osc][interval * s.n_ss : (interval + 1) * s.n_ss]).real)
        
        # add to the summed response array
        s.summed_fft[interval] = s.summed_fft[interval] + s.every_fft[osc][interval]
        
        # take the amplitude and add to the average amplitude array
        s.AOI_fft_amps[osc] = s.AOI_fft_amps[osc] + np.abs(s.every_fft[osc][interval])
      
      # at the end of each "osc" loop, summed_fft[interval] will have all of the oscillators in there. 
      # We will take the amplitude and at it to our summed_avg_fft_amps array
      s.summed_AOI_fft_amps = s.summed_AOI_fft_amps + np.abs(s.summed_fft[interval])

    # divide by # intervals to get final average
    s.AOI_fft_amps = s.AOI_fft_amps / s.num_intervals
    s.summed_AOI_fft_amps = s.summed_AOI_fft_amps / s.num_intervals





  def save(s, filename = None):
    """ Saves your vodscillator in a .pkl file
 
    Parameters
    ------------
        filename: string, Optional
          Don't include the ".pkl" at the end of the string!
          If no filename is provided, it will just use the "name" given to your vodscillator
        
    """
    
    if filename:
      f = filename + ".pkl"
    else:
      f = s.name + ".pkl"

    with open(f, 'wb') as outp:  # Overwrites any existing file with this filename!.
        pickle.dump(s, outp, pickle.HIGHEST_PROTOCOL)

  def __str__(s):
    return f"A vodscillator named {s.name} with {s.num_osc} oscillators!"
  
  def ODE(s, t, z):
    # This function will only be called by the ODE solver

    # Note that this is equation (11) in V&D 2008

    # First make an array to represent the current (complex) derivative of each oscillator
    ddt = np.zeros(s.num_osc, dtype=complex)

    # Define the complex coupling constant (ccc)
    ccc =  s.d_R + 1j*s.d_I

    for k in range(0, s.num_osc - 1):
      # This "universal" part of the equation is the same for all oscillators. 
      # (Note our xi are functions of time, and z[k] is the current position of the k-th oscillator)
      universal = (1j*s.omegas[k]* + s.epsilon)*z[k] + s.xi_glob(t) + s.xi_loc[k](t) - s.B*((np.abs(z[k]))**2)*z[k]

      # COUPLING

      # if we're in the middle of the chain, we couple with the oscillator on either side
      if k != 0 & k != (s.num_osc - 1):
        ddt[k] = universal + ccc*((z[k+1] - z[k]) + (z[k-1] - z[k]))

      # But if we're at an endpoint, we only have one oscillator to couple with
      elif k == 0:
        ddt[k] = universal + ccc*(z[k+1] - z[k])
      elif k == s.num_osc - 1:
        ddt[k] = universal + ccc*(z[k-1] - z[k])
      
    return ddt
  
  #NOW WE PLOT!

  def plot_waveform(s, osc = -1, component = "re", ss = False, xmin = 0.0, xmax = None, ymin = None, ymax = None, fig_num = 1, ):
    """ Plots a waveform for a given oscillator
 
    Parameters
    ------------
        index: int, Optional
          The index of your oscillator (-1 gives summed response)
        component: str, Optional
          Which component of signal to plot; "re" or "im" for real or imaginary, respectively
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

    plt.figure(fig_num)
    plt.plot(t, y)
    plt.show()


  def plot_spectra(s, osc = -1, interval = -1, xmin = -0.1, xmax = None, ymin = 0.0, ymax = None, fig_num = 1):

    """ Plots the fft waveform for a given oscillator (or summed response) in steady state
 
    Parameters
    ------------
        index: int = -1
          The index of your oscillator (-1 gives summed response)
        fig_num: int, Optional
          Only required if plotting multiple figures
        xmin: float, Optional
        xmax: float, Optional
        ymin: float, Optional
        ymax: float, Optional
        
    """

    f = s.fft_freq

    if osc == -1:
      if interval == -1:  
        y = s.summed_AOI_fft_amps
      else:
        y = np.abs(s.summed_fft[interval])
    else:
      if interval == -1:
        y = s.AOI_fft_amps[osc]
      else:
        y = np.abs(s.every_fft[osc][interval])

    plt.figure(fig_num)
    plt.plot(f, y)
    plt.xlim(left = xmin)
    plt.xlim(right = xmax)
    plt.ylim(bottom = ymin)
    plt.ylim(top = ymax)
    plt.show()

  def plot_freq_clusters(s, fig_num = 1):
    """ Creates V&D style frequency clustering plots
    Parameters
    ------------
        fig_num: int, Optional
          Only required if plotting multiple figures
    """

    # first, we get our curve of characteristic frequencies
    s.char_freqs = s.omegas / (2*np.pi)

    # next, we get our "average position amplitudes" (square root of the average of the square of the real part of z)
    s.avg_position_amplitudes = np.zeros(s.num_osc)
    # and the average frequency of each oscillator
    s.avg_cluster_freqs = np.zeros(s.num_osc)

    for osc in range(s.num_osc):
      s.avg_position_amplitudes[osc] = np.sqrt(np.mean((s.ss_sol[osc].real)**2))
     
      # we're going to (temporarily) exclude the peak around 0 freq here!
      min_freq_pt = int(1 * s.h)

      s.avg_cluster_freqs[osc] = np.mean(s.AOI_fft_amps[osc][min_freq_pt:] / (2*np.pi))
    
    # now plot!
    plt.plot(s.avg_cluster_freqs, '-o', label="Average frequency")
    plt.plot(s.avg_position_amplitudes, label="Amplitude")
    plt.plot(s.char_freqs, '--', label="Characteristic frequency")
    plt.ylabel('Frequency')
    plt.xlabel('Oscillator')
    plt.title(f"Frequency clustering, sigma local = {s.loc_noise_amp}, sigma global = {s.glob_noise_amp}")
    plt.legend()
    plt.show()






    

