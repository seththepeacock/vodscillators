import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq

class Vodscillator:
  """
  Vod-structions
  1. Create Vodscillator
  2. Initialize frequency distribution, ICs, and (optional) non-isochronicity with "initialize(**p)"
  3. Generate Noise with "gen_noise(**p)"
  4. Pass in ODE parameters and solve ODE Function with "solve_ODE(**p)" 
  5. Do FFT with "do_fft()"
  6. Save (pickle) your vodscillator to a file with "save()"
  7. Plot!
  """

  def __init__(s, **p):
    # s = self (refers to the object itself)
    # **p unpacks the dictionary of parameters (p) we pass into the initializer

    # GENERAL PARAMETERS
    s.num_osc = p["num_osc"]  # number of oscillators in chain [default = 100 or 150]
    
    if "name" in p:
      s.name = p["name"] # name your vodscillator!

  def initialize(s, **p):
    """
    Generates frequency distribution (omegas[]), initial conditions (ICs[]), and betas[]
    """

    # NECESSARY PARAMETERS
    s.freq_dist = p["freq_dist"] # set to "linear" for frequency increasing linearly, set to "exp" for frequency increasing exponentially
    s.roughness_amp = p["roughness_amp"] # variation in each oscillator's characteristic frequency
    s.omega_0 = p["omega_0"]  # char frequency of lowest oscillator [default = 2*np.pi] 
    s.omega_N = p["omega_N"]  # char frequency of highest oscillator [default = 5*(2*np.pi)] 
    s.IC_method = p["IC_method"]  # set to "rand" for randomized initial conditions, set to "const" for constant initial conditions
    s.beta_sigma = p["beta_sigma"] # standard deviation for imaginary coefficient for cubic nonlinearity

    # Tonotopic Frequency Distribution
    # Now we set the frequencies of each oscillator in our chain - linear or exponential
    if s.freq_dist == "linear":
      s.omegas = np.linspace(s.omega_0,s.omega_N,s.num_osc) # linearly spaced frequencies from omega_0 to omega_N
    elif s.freq_dist == "exp":
      s.omegas = np.zeros(s.num_osc, dtype=float)
      for k in range(s.num_osc): # exponentially spaced frequencies from omega_0 to omega_N
        A = s.omega_0
        B = (s.omega_N/s.omega_0)**(1/(s.num_osc - 1))
        s.omegas[k] = A*(B**k)
        # note that omegas[0] = A = omega_0, and omegas[num_osc - 1] = A * B = omega_N 

    # add roughness to the freq distribution
    r = s.roughness_amp
    roughness = np.random.uniform(low=-r, high=r, size=(s.num_osc,))
    s.omegas = s.omegas + roughness

    # Generate initial conditions and store in ICs[]
    s.ICs = np.zeros(s.num_osc, dtype=complex)

    if s.IC_method == "rand":
      # generate random ICs
      for k in range(s.num_osc):
        x_k = np.random.uniform(-1, 1)
        y_k = np.random.uniform(-1, 1) # this was (0,1) in Beth's code
        s.ICs[k] = complex(x_k, y_k) # make a complex combination of x and y and save it in ICs
    elif s.IC_method == "const":
      # generate the same predetermined ICs as Beth
      all_x = np.linspace(-1, 1, s.num_osc)
      all_y = np.linspace(1, -1, s.num_osc)
      for k in range(s.num_osc):
        x_k = all_x[k]
        y_k = all_y[k]
        s.ICs[k] = x_k - (y_k*1j/s.omegas[k])

    # generate beta_j using a gaussian centered at 0 with std deviation beta_sigma (as in Faber & Bozovic)
    s.betas = np.random.normal(loc=0.0, scale=s.beta_sigma, size=s.num_osc)


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
    s.alpha = p["alpha"] # [default = 1.0] --> real coefficient for cubic nonlinearity

    # Define complex coupling coefficient ccc
    s.ccc = s.d_R + 1j*s.d_I

    # Numerically integrate our ODE from ti to tf with sample rate 1/h
    s.tpoints = np.arange(s.ti, s.tf, s.h) # array of time points
    s.sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y
    # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
    # so s.sol[2, 1104] is the value of the solution for the 3rd oscillator at the 1105th time point.

    # Now get the summed response of all the oscillators (SOO = Summed Over Oscillators)
    s.SOO_sol = np.zeros(len(s.tpoints), dtype=complex)
    for k in range(s.num_osc):
      s.SOO_sol += s.sol[k]

    # It will also be useful to have versions of these solutions restricted to after the system has entered steady state (ss).
    s.ss_sol = s.sol[:, s.n_transient:]
    s.SOO_ss_sol = s.SOO_sol[s.n_transient:]

  def ODE(s, t, z):
    # This function will only be called by the ODE solver

    # Mark the current point in time to track progress
    print(f"Time = {int(t)}/{int(s.tf)}")

    # First make an array to represent the current (complex) derivative of each oscillator
    ddt = np.zeros(s.num_osc, dtype=complex)

    # We are using equation (11) in Vilfan & Duke 2008
    for k in range(s.num_osc):
      # This "universal" part of the equation is the same for all oscillators. 
      # (Note our xi are functions of time, and z[k] is the current position of the k-th oscillator)
      universal = (1j*s.omegas[k] + s.epsilon)*z[k] + s.xi_glob(t) + s.xi_loc[k](t) - (s.alpha + s.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]

      # COUPLING

      # if we're at an endpoint, we only have one oscillator to couple with
      if k == 0:
        ddt[k] = universal + s.ccc*(z[k+1] - z[k])
      elif k == s.num_osc - 1:
        ddt[k] = universal + s.ccc*(z[k-1] - z[k])
      # but if we're in the middle of the chain, we couple with the oscillator on either side
      else:
        ddt[k] = universal + s.ccc*((z[k+1] - z[k]) + (z[k-1] - z[k]))

    return ddt

  def do_fft(s):
    """ Returns four arrays:
    1. every_fft[oscillator index, ss interval index, output]
    2. SOO_fft[output]
    3. AOI_fft[oscillator index, output]
    4. SOO_AOI_fft[output]

    AOI = Averaged Over Intervals (for noise)

    """
    # first, we get frequency axis: the # of frequencies the fft checks depends on the # signal points we give it (n_ss), 
    # and sample spacing (h) tells it what these frequencies correspond to in terms of real time 
    s.fft_freq = rfftfreq(s.n_ss, s.h)
    s.num_freq_points = len(s.fft_freq)
    
    # compute the (r)fft for all oscillators individually and store them in "every_fft"
      # note we are taking the r(eal)fft since (presumably) we don't lose much information by only considering the real part (position) of the oscillators  
    s.every_fft = np.zeros((s.num_osc, s.num_intervals, s.num_freq_points), dtype=complex) # every_fft[osc index, which ss interval, fft output]

    for interval in range(s.num_intervals):
      for osc in range(s.num_osc):
        # calculate fft
        n_start = interval * s.n_ss
        n_stop = (interval + 1) * s.n_ss
        s.every_fft[osc, interval, :] = rfft((s.ss_sol[osc, n_start:n_stop]).real)

    # we'll add them all together to get the fft of the summed response (sum of fft's = fft of sum)
    s.SOO_fft = np.sum(s.every_fft, 0)

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

  

    

  






    
