import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp


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
  7. Save your vodscillator to a file
  7. Plot
  """

  #Note - you must create a frequency distribution before setting ICs since an oscillator's IC depends on its char freq


  def __init__(s, **p):
    # s = self (refers to the object itself)
    # **p unpacks the dictionary of parameters (p) we pass into the initializer

    #if we want to initialize using a file:
    if "filename" in p:
      s.open(p["filename"])
    #otherwise
    else:
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
    # SHOULD WE ADD IMAGINARY NOISE
    s.ti = p["ti"] # start time; [default = 0]
    s.n_transient = p["n_transient"]  # the # of time points we give for transient behavior to settle down; around 30000 [default = 35855]
    s.n_ss = p["n_ss"]  # the # of time points we observe the steady state behavior for [default = 8192]
    s.num_runs = p["num_runs"] # [default for no noise is 1; with noise we must average over multiple runs]
    s.sample_rate = p["sample_rate"] # [default = 128]

    # Calculate other params
    s.h = 1/s.sample_rate #delta t between time points
    s.tf = s.h*s.n_transient + s.h*(s.n_ss + 1)  # end time - not sure what this + 1 is for

    # We want a global xi(t) and then one for each oscillator. 

    # First, generate time points with a little padding at the end to smooth the interpolation
    s.padded_t = np.arange(s.ti, s.tf + 2*s.h, s.h)

    # global --> will impact each oscillator equally at each point in time (e.g., wind blowing?)
    # first we randomly generate points with a standard normal distribution
    global_noise = s.glob_noise_amp * np.random.randn(len(s.padded_t)) # normal distribution times noise amplitude
    # then interpolate between (using a cubic spline) for ODE solving adaptive step purposes
    s.xi_glob = CubicSpline(s.padded_t, global_noise)

    # local --> will impact each oscillator differently at each point in time (e.g., brownian motion of fluid in inner ear surrounding hair cells)
    s.xi_loc = np.empty(s.num_osc, dtype=CubicSpline)
    for k in range(s.num_osc):
      # again, we randomly generate points (standarad normal) then interpolate between
      local_noise = s.loc_noise_amp * np.random.randn(len(s.padded_t))
      s.xi_loc[k] = CubicSpline(s.padded_t, local_noise)


  def set_ODE(s, **p):
    # Setting Parameters for Final Function 

    # NECESSARY PARAMETERS
    s.epsilon = p["epsilon"] # [default = 1.0] --> control parameter
    s.d_R = p["d_R"]  # [default = 0.15] --> real part of coupling coefficient
    s.d_I = p["d_I"]  # [default = -1.0] --> imaginary part of coupling coefficient
    s.B = p["B"] # [default = 1.0] --> amount of cubic nonlinearity

  def solve_ODE(s, **p):
    s.t = np.arange(s.ti, s.tf, s.h) # array of time points
    s.sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.t).y 
    # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
    # so s.sol[2][1104] is the value of the solution for the 3rd oscillator at the 1105th time point.

    s.summed_sol = np.zeros(len(s.t))
    for k in range(s.num_osc):
      s.summed_sol = s.summed_sol + s.sol[k]
    

  def save(s, filename):
    # filename must be a string like "file.pkl"
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
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
      # This part of the equation is the same for all oscillators. 
      # (Note our xi are functions of time, and z[k] is the current position of the k-th oscillator)
      # (Note also the t+1 argument in the xi's are so that we never use the endpoints of the interpolation, which we had padded with 2 extra points)
      universal = (1j*s.omegas[k]* + s.epsilon)*z[k] + s.xi_glob(t+1) + s.xi_loc[k](t+1) - s.B*((np.abs(z[k]))**2)*z[k]


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

  def plot_waveform(s, index, component, fig_num):
    if index == "sum":
      y = s.summed_sol
    else:
      y = s.sol(index)

    if component == "im":
      y = y.imag
    elif component == "re":
      y = y.real
    
    plt.figure(fig_num)
    plt.plot(s.t, y)
    plt.show()



    

