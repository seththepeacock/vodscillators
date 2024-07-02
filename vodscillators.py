import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq, rfft

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
    s.n_ss = p["n_ss"]  # the # of time points we observe the steady state behavior for [default = 8192]
    s.num_runs = p["num_runs"] # [default for no noise is 1; with noise we must average over multiple runs]
    s.sample_rate = p["sample_rate"] # [default = 128]

    # Calculate other params
    s.h = 1/s.sample_rate #delta t between time points
    s.tf = s.h*s.n_transient + s.h*(s.n_ss)  # end time

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


  def set_ODE(s, **p):
    # Setting Parameters for Final Function 

    # NECESSARY PARAMETERS
    s.epsilon = p["epsilon"] # [default = 1.0] --> control parameter
    s.d_R = p["d_R"]  # [default = 0.15] --> real part of coupling coefficient
    s.d_I = p["d_I"]  # [default = -1.0] --> imaginary part of coupling coefficient
    s.B = p["B"] # [default = 1.0] --> amount of cubic nonlinearity

  def solve_ODE(s):
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

  def get_fft(s, **p):
    # first, we want to restrict our solution to after the system has entered steady state (ss).
    # we generate an array which is like our solution array, except with only timepoints after n_transient
    s.ss_sol = s.sol[:, s.n_transient:]

    
    #get frequency axis (depends on # signal points n_ss and sample spacing h)
    s.fft_freq = fftfreq(s.n_ss, s.h)
    
    # compute the fft for all oscillators individually and store them in "every_fft"
    s.every_fft = np.zeros((s.num_osc, s.num_runs, len(s.fft_freq)), dtype=complex)
    for osc in range(s.num_osc):
      for run in range(s.num_runs):
        s.every_fft[osc][run] = fft(s.ss_sol[osc][run * s.n_ss : (run + 1) * s.n_ss])

    #GOAL 1- average over all runs (for each oscillator)
    s.avg_fft_amps = np.zeros((s.num_osc, len(s.fft_freq)), dtype=complex)
    for osc in range(s.num_osc):
      for run in range(s.num_runs):
        s.avg_fft_amps[osc] = np.mean(s.every_fft[osc])

    #done i guess? -deniz

    #GOAL 2- calculate phase difference coherence between each neighboring run (for each oscillator)
    """we should probably store the previous fft's and reuse them here instead of re-doing the fft's here"""    
    for osc in range(s.num_osc):
      Npts = s.n_ss
      #wf is our time series (waveform)
      wf = s.ss_sol[osc]
      M = int(np.floor(len(wf)/Npts)) #previously s.num_runs
      #print(wf.shape)
      SR = s.sample_rate #sampling frequency
      freq= np.arange(0,(Npts+1)/2,1)    # create a freq. array (for FFT bin labeling)
      freq= SR*freq/Npts
      #indxFl= np.where(freq>=200)[0][0]  # find freq index re above (0.2) kHz
      indxFh= np.where(freq<=7000)[0][-1]  # find freq index re under (7) kHz
      indexFl=0

      storeM = np.empty([int(Npts),M])
      storeP = np.empty([int(Npts),M])
      storeWF = np.empty([int(Npts),M])
      storePdiff = np.empty([int(Npts),M-1])  # smaller buffer for phase diffs


# ==  == spectral averaging loop
      for run in range(0, M): #for each run:
          indx = run*Npts  # index offset so to move along waveform
          signal=  np.squeeze(wf[indx:indx+Npts]);  # extract segment
          # --- deal w/ FFT
          spec = fft(signal)  # magnitude of ft of wf
          mag = abs(spec)
          phase = np.angle(spec)
          # --- store away
          #print(len(signal))
          storeM[:,run] = mag #of fourier transform
          storeP[:,run] = phase #of ft
          storeWF[:,run] = signal #the wf ie time series
          # ==== 
          if (run>=1):
              #Q: why not take the last segment from the stored buffers?
              indxL = (run-1)*Npts  # previous segment index 
              signalL =  np.squeeze(wf[indxL:indxL+Npts])  # re-extract last segment
              specL = fft(signalL) 
              phaseL = np.angle(specL)
              # --- now compute phase diff re last segment (phaseDIFF2) and store
              phaseDIFF2 = phase - phaseL
              storePdiff[:,run-1] = phaseDIFF2

      # ====
      # vC= sqrt(mean(sin(phiC-phi0)).^2 +mean(cos(phiC-phi0)).^2);
      xx= np.average(np.sin(storePdiff),axis=1)
      yy= np.average(np.cos(storePdiff),axis=1)
      coherence= np.sqrt(xx**2 + yy**2)


      #n.b. for both goals: average over windows for each osc and for sum of the fft's

      #remains to do everything for sums of fft's and to plot stuff
      #put plot and averages in another function
      SR = s.sample_rate
      freq = s.fft_freq

      tP = np.arange(indx/SR,(indx+Npts-0)/SR,1/SR); # time assoc. for segment (only for plotting)
      specAVGm= np.average(storeM,axis=1)  # spectral-avgd MAGs
      specAVGp= np.average(storeP,axis=1)  # spectral-avgd PHASEs
      specAVGpd= np.average(storePdiff,axis=1) # spectral-avgd phase diff.
      # --- time-averaged version
      timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
      specAVGwf= rfft(timeAVGwf)

      indxFl= np.where(freq>=200)[0][0]  # find freq index re above (0.2) kHz
      indxFh= np.where(freq<=7000)[0][-1]  # find freq index re under (7) kHz

      plt.close("all")
      # --- single waveform and assoc. spectrum
      if 1==0:
          fig1, ax1 = plt.subplots(2,1)
          ax1[0].plot(tP,signal,'k-',label='wf')
          ax1[0].set_xlabel('Time [s]')  
          ax1[0].set_ylabel('Signal [arb]') 
          ax1[0].set_title('Last waveform used for averaging')
          ax1[0].grid()
          ax1[1].plot(freq/1000,20*np.log10(spec),'r-',label='X')
          ax1[1].set_xlabel('Frequency [kHz]')  
          ax1[1].set_ylabel('Magnitude [dB]') 
          ax1[1].set_title('Spectrum')
          ax1[1].grid()
          ax1[1].set_xlim([0,8])
          fig1.tight_layout(pad=1.5)

      # --- averaged spect. MAG
      specAVGmDB= 20*np.log10(specAVGm)
      fig2 = plt.subplots()
      fig2= plt.plot(freq/1000,specAVGmDB,'b-',lw=1,label='X')
      fig2= plt.xlabel('Frequency [kHz]')
      fig2= plt.ylabel('Magnitude [dB]') 
      fig2= plt.title("averaged spectral magnitude") 
      fig2= plt.grid()
      fig2= plt.xlim([0, 7])
      #fig2= plt.ylim([np.min(specAVGmDB[indxFl:indxFh])-5,
      #                np.max(specAVGmDB[indxFl:indxFh])+5])


      # --- averaged time-averaged. MAG
      if 1==0:
          specAVGwfDB= 20*np.log10(abs(specAVGwf))
          fig3 = plt.subplots()
          fig3= plt.plot(freq/1000,specAVGwfDB,'b-',lw=2)
          fig3= plt.xlim([0, 7])
          #fig3= plt.ylim([np.min(specAVGwfDB[indxFl:indxFh])-5,
          #                np.max(specAVGwfDB[indxFl:indxFh])+5])
          fig3= plt.xlabel('Frequency [kHz]')
          fig3= plt.ylabel('Magnitude [dB]') 
          fig3= plt.title('Time-averaged spectrum') 
          fig3= plt.grid()

      """# --- averaged spect. PHASE (difference re lower freq bin)
      # NOTE: Not sure this is coded to completion to ascertain this/that
      if 1==0:
          fig4 = plt.subplots()
          fig4= plt.plot(freq[0:-1]/1000,specAVGpd,'b-',lw=1,label='X')
          #fig4= plt.plot(freq/1000,specAVGp,'b-',lw=1,label='X')
          fig4= plt.xlabel('Frequency [kHz]')  
          fig4= plt.ylabel('Phase [rads]') 
          fig4= plt.title(fileN) 
          fig4= plt.grid()
          fig4= plt.xlim([0, 7])
          #fig2= plt.ylim([-0.2,0.2])"""

      # --- coherence
      if 1==1:
          fig1 = plt.subplots()
          fig1= plt.plot(freq/1000,coherence,'b-',lw=1,label='X')
          #fig4= plt.plot(freq/1000,specAVGp,'b-',lw=1,label='X')
          fig5= plt.xlabel('Frequency [kHz]')  
          fig5= plt.ylabel('Phase Coherence (i.e., vector strength)') 
          fig5= plt.title("coherence") 
          fig5= plt.grid()
          fig5= plt.xlim([0, 7])
          #fig2= plt.ylim([-0.2,0.2])

      plt.show()


  
    
    
    

  def save(s, filename = None):
    """ Saves your vodscillator
 
    Parameters
    ------------
        filename: string, Optional
          Don't include the ".pkl" at the end!
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

  def plot_waveform(s, index, component = "re", fig_num = 1, ss = False):
    """ Plots a waveform for a given oscillator
 
    Parameters
    ------------
        index: int
          The index of your oscillator (-1 gives summed response)
        ss: boolean, Optional
          If you only want the steady state part of the solution
        component: str, Optional
          Which component of signal to plot; "re" or "im" for real or imaginary, respectively
        fig_num: int, Optional
          Only required if plotting multiple figures
        
    """
    if index == -1: #because -1 means "sum"
      y = s.summed_sol
    else:
      y = s.sol[index]

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



    

