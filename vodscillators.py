import numpy as np
from scipy import interpolate


class Vodscillator:

  def __init__(s, **p):
    # s = self (refers to the object itself)
    # **p unpacks the dictionary of parameters (p) we pass into the initializer
    
    s.name = p["name"] #name your vodscillator!

    #set parameters

    s.IC_method = p["IC_method"] # set to "rand" for randomized initial conditions, set to "const" for constant initial conditions
    s.freq_dist = p["freq_dist"] # set to "linear" for linear frequency distribution, set to "exp" for exponential frequency distribution
    s.loc_noise_amp = p["loc_noise_amp"] #amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
    s.glob_noise_amp = p["glob_noise_amp"] #amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
    s.epsilon = p["epsilon"] # [default = 1.0] --> damping
    s.omega_0 = p["omega_0"]  # char frequency of loomegaest oscillator [default = 2*np.pi] 
    s.omega_n = p["omega_n"]  # char frequency of highest oscillator [default = 5*(2*np.pi)] 
    s.d_R = p["d_R"]  # [default = 0.15]
    s.d_I = p["d_I"]  # [default = -1.0] 
    s.B = p["B"] # [default = 1.0]
    s.num_osc = p["num_osc"]  # number of oscillators in chain[default = 100 or 150], 80 in paper but didn't work when Olha tried --> number of oscillators
    s.n_transient = p["n_transient"]  # the # of time points we give for transient behavior to settle down; around 30000 [default = 35855]
    s.n_ss = p["n_ss"]  # the # of time points we observe the steady state behavior for [default = 8192]
    s.num_runs = p["num_runs"] # [default for no noise is 1; when we have noise we average over multiple runs]
    
    #calculate the rest
    
    s.sample_rate = 128  #[default = 128] 
    s.h = 1/s.sample_rate
    s.ti = 0 # start time, [default = 0]
    s.tf = s.h*s.n_transient + s.h*(s.n_ss + 1)  # end time
    s.tpoints = np.arange(s.ti, s.tf, s.h) # array of time points
    s.N = len(s.tpoints) #total num of points
    
    #get initial conditions

    #start with the frequency distribution array (omega)
    #IMPLEMENT ROUGHNESS

    if s.freq_dist == "linear":
      s.omega = np.linspace(s.omega_0,s.omega_n,s.num_osc)
    if s.freq_dist == "exp":
      s.omega = np.zeros(s.num_osc, dtype=float)
      for index in np.arange(0, s.num_osc):
        s.omega[index] = s.omega_0*(s.omega_n/s.omega_0)**(index / (s.num_osc - 1))
        #CHECK THIS IS CORRECT



    if s.IC_method == "rand":
      s.omega = 3
    
  def __str__(s):
    return f"A vodscillator named {s.name} with {s.num_osc} oscillators!"


    

