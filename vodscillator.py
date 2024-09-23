import numpy as np
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
    Generates frequency distribution omegas[], initial conditions ICs[], imaginary nonlinearity coefficients betas[], and sets other ODE parameters 
    """

    # NECESSARY PARAMETERS
    s.IC_method = p["IC_method"]  # set to "rand" for randomized initial conditions, set to "const" for constant initial conditions
    
    s.freq_dist = p["freq_dist"] # set to "linear" for frequency increasing linearly, set to "exp" for frequency increasing exponentially
    s.omega_0 = p["omega_0"]  # char frequency of lowest oscillator [default = 2*np.pi] 
    s.omega_N = p["omega_N"]  # char frequency of highest oscillator [default = 5*(2*np.pi)] 
    
    # if no noise, these will be just be the values for all oscillators
    s.epsilon = p["epsilon"] # [default = 1.0] --> control parameter
    s.alpha = p["alpha"] # [default = 1.0] --> real coefficient for cubic nonlinearity
    s.d_R = p["d_R"]  # [default = 0.15] --> real part of coupling coefficient
    s.d_I = p["d_I"]  # [default = -1.0] --> imaginary part of coupling coefficient

    # these parameters will be the *percent* of noise (np = noise percent) in each oscillators parameters... that is:
      # epsilons[i] = random number uniformly pulled from the range [epsilon +/- epsilon_np*epsilon]
    s.epsilon_np = p["epsilon_np"] 
    s.d_R_np = p["d_R_np"]
    s.d_I_np = p["d_I_np"]
    s.alpha_np = p["alpha_np"]
    s.omega_np = p["omega_np"]
    
    s.beta_sigma = p["beta_sigma"] # standard deviation for imaginary coefficient for cubic nonlinearity (always mean 0)


    # INITIALIZE
    
    # Tonotopic Frequency Distribution
    # Now we set the frequencies of each oscillator in our chain - linear or exponential
    if s.freq_dist == "linear":
      s.omegas = np.linspace(s.omega_0,s.omega_N,s.num_osc) # linearly spaced frequencies from omega_0 to omega_N
    elif s.freq_dist == "exp":
      s.omegas = np.zeros(s.num_osc, dtype=float)
      for k in range(s.num_osc): # exponentially spaced frequencies from omega_0 to omega_N
        a = s.omega_0
        b = (s.omega_N/s.omega_0)**(1/(s.num_osc - 1))
        s.omegas[k] = a*(b**k)
          # note that omegas[0] = A = omega_0, and omegas[num_osc - 1] = A * B = omega_N 
        
    # add roughness
    for k in range(s.num_osc):
      noise_amp = s.omegas[k]*s.omega_np
      s.omegas[k] = np.random.uniform(s.omegas[k] - noise_amp, s.omegas[k] + noise_amp)


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
        s.ICs[k] = x_k + (y_k*1j) # this was x_k - (y_k*1j/s.omegas[k]) in Beth's code

    # generate actual parameters (if no noise, these will be constant)
    amp = s.epsilon*s.epsilon_np
    s.epsilons = np.random.uniform(s.epsilon - amp, s.epsilon + amp, s.num_osc)
    amp = s.alpha*s.alpha_np
    s.alphas = np.random.uniform(s.alpha - amp, s.alpha + amp, s.num_osc)
    amp = s.d_R*s.d_R_np
    s.d_Rs = np.random.uniform(s.d_R - amp, s.d_R + amp, s.num_osc)
    amp = s.d_I*s.d_I_np
    s.d_Is = np.random.uniform(s.d_I - amp, s.d_I + amp, s.num_osc)
    
    # generate beta_j using a gaussian centered at 0 with std deviation beta_sigma (as in Faber & Bozovic)
    s.betas = np.random.normal(loc=0.0, scale=s.beta_sigma, size=s.num_osc)

    # Define array of complex coupling coefficients ccc's (will be used in ODE)
    s.cccs = s.d_Rs + 1j*s.d_Is

  def gen_noise(s, **p):
    # Generating Noise - creates s.xi_glob and s.xi_loc[]

    # NECESSARY PARAMETERS
    s.loc_noise_amp = p["loc_noise_amp"] # amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
    s.glob_noise_amp = p["glob_noise_amp"] # amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
    s.sample_rate = p["sample_rate"] # [default = 128]
    s.ti = p["ti"] # start time; [default = 0]
    s.t_transient = p["t_transient"] # how long we give for transient behavior to settle down [default = 280 --> n.transient = 35840]
    s.t_win = p["t_win"] # length of an win of ss observation [default = 64 --> n.transient = 8192]
    s.num_wins = p["num_wins"] # [default for no noise is 1; with noise we must average over multiple wins]


    # Calculate other params
    s.delta_t = 1/s.sample_rate #delta t between time points
    s.n_transient = s.t_transient * s.sample_rate # num of time points corresponding to t_transient
    s.n_win = s.t_win * s.sample_rate
    s.tf = s.t_transient + s.num_wins * s.t_win
    
    # We want a global xi(t) and then one for each oscillator. 

    # First, generate time points
    s.tpoints = np.arange(s.ti, s.tf, s.delta_t)

    # global --> will impact all hair bundles (and associated papilla for TwinVods) equally
    
    # first we randomly generate points uniformly within the given amplitude range
    global_noise = np.random.uniform(-s.glob_noise_amp, s.glob_noise_amp, len(s.tpoints)) 
    # then interpolate between (using a cubic spline) for ODE solving adaptive step purposes
    s.xi = CubicSpline(s.tpoints, global_noise)

    # local --> will impact each oscillator differently at each point in time (e.g., brownian motion of fluid in inner ear surrounding hair cells)
    s.xi_loc = np.empty(s.num_osc, dtype=CubicSpline)
    for k in range(s.num_osc):
      local_noise = np.random.uniform(-s.loc_noise_amp, s.loc_noise_amp, len(s.tpoints))
      s.xi_loc[k] = CubicSpline(s.tpoints, local_noise)


  def solve_ODE(s):
    # Numerically integrate our ODE from ti to tf with sample rate 1/h
    s.tpoints = np.arange(s.ti, s.tf, s.delta_t) # array of time points
    s.sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y
    # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
    # so s.sol[2, 1104] is the value of the solution for the 3rd oscillator at the 1105th time point.

    # Now get the summed response of all the oscillators (SOO = Summed Over Oscillators)
    s.SOO_sol = np.sum(s.sol, 0)

  def ODE(s, t, z):
    # This function will only be called by the ODE solver
    
    # Mark the current point in time to track progress
    if np.abs(t - int(t)) < 0.01:
      print(f"Time = {int(t)}/{int(s.tf)}")

    # First make an array to represent the current (complex) derivative of each oscillator
    ddt = np.zeros(s.num_osc, dtype=complex)

    # We are using equation (11) in Vilfan & Duke 2008 for the derivatives
    for k in range(s.num_osc):
      # This "universal" part of the equation is the same for all oscillators. 
      # (Note our xi are functions of time, and z[k] is the current position of the k-th oscillator)
      universal = (1j*s.omegas[k] + s.epsilons[k])*z[k] + s.xi(t) + s.xi_loc[k](t) - (s.alphas[k] + s.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]

      # ADD COUPLING

      # if we're at an endpoint, we only have one oscillator to couple with
      if k == 0:
        ddt[k] = universal + s.cccs[k]*(z[k+1] - z[k])
      elif k == s.num_osc - 1:
        ddt[k] = universal + s.cccs[k]*(z[k-1] - z[k])
        
      # but if we're in the middle of the chain, we couple with the oscillator on either side
      else:
        ddt[k] = universal + s.cccs[k]*((z[k+1] - z[k]) + (z[k-1] - z[k]))

    return ddt

  def do_fft(s):
    """ Returns four arrays:
    1. every_fft[oscillator index, ss win index, output]
    2. SOO_fft[output]
    3. AOI_fft[oscillator index, output]
    4. SOO_AOI_fft[output]

    AOI = Averaged Over Wins (for noise)

    """
    # first, we get frequency axis: the # of frequencies the fft checks depends on the # signal points we give it (n_win), 
    # and sample spacing (h) tells it what these frequencies correspond to in terms of real time 
    s.fft_freq = rfftfreq(s.n_win, s.delta_t)
    s.num_freq_points = len(s.fft_freq)
    
    # compute the (r)fft for all oscillators individually and store them in "every_fft"
      # note we are taking the r(eal)fft since (presumably) we don't lose much information by only considering the real part (position) of the oscillators  
    s.every_fft = np.zeros((s.num_osc, s.num_wins, s.num_freq_points), dtype=complex) # every_fft[osc index, which ss win, fft output]

    # truncate to the ss solution (all points after n_transient)
    ss_sol = s.sol[:, s.n_transient:]
    for win in range(s.num_wins):
      for osc in range(s.num_osc):
        # calculate fft
        n_start = win * s.n_win
        n_stop = (win + 1) * s.n_win
        s.every_fft[osc, win, :] = rfft((ss_sol[osc, n_start:n_stop]).real)

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
  
  def params(s, paramlist=None):
    params = vars(s)
    entries_to_remove = ('omegas', 'ICs', 'betas', 'tpoints', 'xi_glob', 'xi_loc', 'sol', 'SOO_sol', 'fft_freq', 'every_fft', 'SOO_fft')
    for k in entries_to_remove:
      params.pop(k, None)
    if paramlist != None:
      r = dict()
      for k in paramlist:
        r[k] = params[k]
      return r
    return params



            

            


      





    

