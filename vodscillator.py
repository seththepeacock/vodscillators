import numpy as np
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from itertools import combinations
   

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
    s.freq_dist = p["freq_dist"] # set to "linear" for frequency increasing linearly, set to "exp" for frequency increasing exponentially
    s.roughness_amp = p["roughness_amp"] # variation in each oscillator's characteristic frequency
    s.omega_0 = p["omega_0"]  # char frequency of lowest oscillator [default = 2*np.pi] 
    s.omega_N = p["omega_N"]  # char frequency of highest oscillator [default = 5*(2*np.pi)] 
    s.IC_method = p["IC_method"]  # set to "rand" for randomized initial conditions, set to "const" for constant initial conditions
    s.beta_sigma = p["beta_sigma"] # standard deviation for imaginary coefficient for cubic nonlinearity
    s.epsilon = p["epsilon"] # [default = 1.0] --> control parameter
    s.d_R = p["d_R"]  # [default = 0.15] --> real part of coupling coefficient
    s.d_I = p["d_I"]  # [default = -1.0] --> imaginary part of coupling coefficient
    s.alpha = p["alpha"] # [default = 1.0] --> real coefficient for cubic nonlinearity

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

    # Define complex coupling coefficient ccc (will be used in ODE)
    s.ccc = s.d_R + 1j*s.d_I

  def gen_noise(s, **p):
    # Generating Noise - creates s.xi_glob and s.xi_loc[]

    # NECESSARY PARAMETERS
    s.loc_noise_amp = p["loc_noise_amp"] # amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
    s.glob_noise_amp = p["glob_noise_amp"] # amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
    s.sample_rate = p["sample_rate"] # [default = 128]
    s.ti = p["ti"] # start time; [default = 0]
    s.t_transient = p["t_transient"] # how long we give for transient behavior to settle down [default = 280 --> n.transient = 35840]
    s.t_ss = p["t_ss"] # length of an interval of ss observation [default = 64 --> n.transient = 8192]
    s.num_intervals = p["num_intervals"] # [default for no noise is 1; with noise we must average over multiple intervals]


    # Calculate other params
    s.delta_t = 1/s.sample_rate #delta t between time points
    s.n_transient = s.t_transient * s.sample_rate # num of time points corresponding to t_transient
    s.n_ss = s.t_ss * s.sample_rate # num of time points corresponding to t_ss
    s.tf = s.t_transient + s.num_intervals * s.t_ss
    
    # We want a global xi(t) and then one for each oscillator. 

    # First, generate time points
    s.tpoints = np.arange(s.ti, s.tf, s.delta_t)

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
    s.fft_freq = rfftfreq(s.n_ss, s.delta_t)
    s.num_freq_points = len(s.fft_freq)
    
    # compute the (r)fft for all oscillators individually and store them in "every_fft"
      # note we are taking the r(eal)fft since (presumably) we don't lose much information by only considering the real part (position) of the oscillators  
    s.every_fft = np.zeros((s.num_osc, s.num_intervals, s.num_freq_points), dtype=complex) # every_fft[osc index, which ss interval, fft output]

    # truncate to the ss solution (all points after n_transient)
    ss_sol = s.sol[:, s.n_transient:]
    for interval in range(s.num_intervals):
      for osc in range(s.num_osc):
        # calculate fft
        n_start = interval * s.n_ss
        n_stop = (interval + 1) * s.n_ss
        s.every_fft[osc, interval, :] = rfft((ss_sol[osc, n_start:n_stop]).real)

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


  def analytic_phase_coherence(s, cluster_width=0.005, f_min=0.0, f_max=10.0, f_resolution=0.001, num_wins=100, t_win=1, amp_weights=True):
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
    # get SS part of solution
    ss_sol = s.sol[:, s.n_transient:]
    #get analytic signals of each oscillator
    analytic_signals = hilbert(ss_sol.real, axis=1)
    # get phases and amps
    inst_phases = np.unwrap(np.angle(analytic_signals))
    inst_amps = np.abs(analytic_signals)
    # get # points in window
    n_win = int(t_win * s.sample_rate)
    # generate frequency array
    s.apc_freqs = np.arange(f_min, f_max, f_resolution) #apc stands for analytic phase coherence
    num_freqs = len(s.apc_freqs)

    # get clusters
    def cluster():
      # take "derivatives" to get instantaenous frequencies
      inst_freqs = (np.diff(inst_phases) / (2.0*np.pi) * s.sample_rate)
      clusters = np.zeros((num_wins, num_freqs, s.num_osc))

      # pick a window
      for win in range(num_wins):
        print(f"Clustering Window {win}")
        # find average frequency for each oscillator
        avg_freqs = np.average(inst_freqs[:, win*n_win:(win+1)*n_win], axis=1)
        # for each frequency box we look through each oscillator to see which oscillators are close enough to that frequency
        for f in range(num_freqs):
          for osc in range(s.num_osc):
            # # keep track of how many oscillators we've found in the cluster
            # osc_in_clusters = 0
            if abs(avg_freqs[osc] - s.apc_freqs[f]) < cluster_width:
                # put the oscillator's index in the ith position
                clusters[win, f, osc] = 1
                # i += 1
      return clusters
    
    # get all clusters
    s.clusters = cluster()
    # initialize array to store all phase coherences
    all_phase_coherences = np.zeros((num_wins, num_freqs))

    for win in range(num_wins):
        for f in range(len(s.apc_freqs)):
          print(f"Window {win}: Finding PC for {s.apc_freqs[f]}Hz")
          # create list of osc_indices in the cluster
          osc_indices = np.where(s.clusters[win, f] == 1)[0]
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
  
    # average over all t_wins
    s.apc = np.mean(all_phase_coherences, 0)
            

            


      





    

