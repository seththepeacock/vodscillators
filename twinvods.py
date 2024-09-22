import numpy as np
import pickle
from vodscillator import *
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq

# implement?
    # 1. random ICs (with specifiable seed) for papillas/IAC


class TwinVods:
    """
    Vod-structions
    1. Create Two Vodscillators
    2. Initialize frequency distribution, ICs, and (optional) non-isochronicity with "initialize(**p)"
    3. Generate Noise with "gen_noise(**p)"
    4. Create your TwinVodscillator

    """
    def __init__(s, vod_L=Vodscillator, vod_R=Vodscillator, **p):
        # s = self (refers to the object itself)
        # **p unpacks the dictionary of parameters (p) we pass into the initializer
        s.vod_L = vod_L #left Vodscillator
        s.vod_R = vod_R #right Vodscillator
        s.name = p["name"] # name your TwinVodscillators! (also will be used as a filename)
        s.OMEGA_P = p["OMEGA_P"] # papilla char freq
        s.EPSILON_P = p["EPSILON_P"] # papilla damping (negative for damped harmonic oscillator)
        s.D_R_P = p["D_R_P"] # papilla dissipative coupling w/ hair bundles (real coefficient) 
        s.D_I_P = p["D_I_P"] # papilla reactive coupling w/ hair bundles (imaginary coefficient) 
        s.D_R_IAC = p["D_R_IAC"] # IAC dissipative coupling w/ papillas (real coefficient) 
        s.D_I_IAC = p["D_I_IAC"] # IAC reactive coupling w/ papillas (imaginary coefficient)
        s.IAC_noise_amp = p["IAC_noise_amp"] # amplitude of (uniformly generated) noise applied just to IAC
        s.glob_glob_noise_amp = p["glob_glob_noise_amp"] # amplitude of (uniformly generated) noise applied to everything in system
        
        # define our complex coupling constants (CCCs)
        s.P_CCC = s.D_R_P + 1j*s.D_I_P
        s.IAC_CCC = s.D_R_IAC + 1j*s.D_I_IAC
        
        # check that our time parameters are the same for both or we might have some issues!
        vod_params = np.empty((2, 6))
        i = 0
        for v in [s.vod_L, s.vod_R]:
            vod_params[i, :] = np.array([v.sample_rate, v.ti, v.tf, v.t_transient, v.t_win, v.num_wins])
            i = 1
            
        # if they agree, set all of the TV's parameters to them
        if all(vod_params[0, :] == vod_params[1, :]):
            print("Vods agree!")
            s.sample_rate = vod_L.sample_rate
            s.ti = vod_L.ti
            s.tf = vod_L.tf
            s.t_transient = vod_L.t_transient
            s.t_ss = vod_L.t_win
            s.n_transient = vod_L.n_transient
            s.n_win = vod_L.n_win
            s.num_wins = vod_L.num_wins
            s.tpoints = vod_L.tpoints
        else:
            print("Parameter mismatch!")

        # generate noise
        global_global_noise = np.random.uniform(-s.glob_glob_noise_amp, s.glob_glob_noise_amp, len(s.tpoints)) 
        s.IAC_noise_amp = np.random.uniform(-s.IAC_noise_amp, s.IAC_noise_amp, len(s.tpoints))
        # then interpolate between (using a cubic spline) for ODE solving adaptive step purposes
        s.xi = CubicSpline(s.tpoints, global_global_noise)
        s.xi_IAC = CubicSpline(s.tpoints, s.IAC_noise_amp)
        # define some useful parameters
        s.total_num_osc = s.vod_L.num_osc + s.vod_R.num_osc
      

    def solve_ODE(s):
        # NOTE WE ALWAYS PUT LEFT BEFORE RIGHT
        
        # create ICs by concatenating left and right IC arrays with zeros for papillas/IAC
        s.ICs = np.concatenate((s.vod_L.ICs, s.vod_R.ICs, np.array([0, 0, 0])))
        
        # Numerically integrate our ODE (making sure that it gives us the value at our tpoints)
        sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y
        
        # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
        # lets add a dimension at the beginning of osc_sol to track if its the left or right ear
            # in case each has different # oscillators, we'll use the max of the two
        max_num_osc = max(s.vod_L.num_osc, s.vod_R.num_osc)
        s.osc_sol = np.zeros((2, max_num_osc, len(s.tpoints)), dtype=complex)
        s.osc_sol[0, :, :] = sol[0:s.vod_L.num_osc, :]
        s.osc_sol[1, :, :] = sol[s.vod_L.num_osc:(s.total_num_osc), :]
           
            # (s.sol[1, 2, 1104] is the value of the solution for the 3rd oscillator in the right ear at the 1105th time point)
       
        # Now get the summed response of all the oscillators (SOO = Summed Over Oscillators)
        s.SOO_L_sol = np.sum(s.osc_sol[0, :, :], 0)
        s.SOO_R_sol = np.sum(s.osc_sol[1, :, :], 0)
        s.P_L_sol = sol[-3]
        s.P_R_sol = sol[-2]
        s.Z_C_sol = sol[-1]

    def ODE(s, t, z):
        # This function will only be called by the ODE solver
        # Mark the current point in time to track progress
        print(f"Time = {int(t)}/{int(s.tf)}")
        # First make an array to represent the current (complex) derivative of each oscillator
        ddt = np.zeros(s.total_num_osc+3, dtype=complex) #last 3 are T_1, T_2 and X_C
        # (We are adapting equation (11) in Vilfan & Duke 2008)

        # GET VARIABLES
        # summed vodscillators
        X_l = np.sum(z[s.vod_L.num_osc:s.vod_R.num_osc])
        X_r = np.sum(z[0:s.vod_R.num_osc])
        # papillas
        P_l = z[-3]
        P_r = z[-2]
        # air cavity
        Z_c = z[-1]
        
        # Now define our derivatives!
        
        # LEFT EAR
        for k in range(s.vod_L.num_osc):
            # The "universal" part of the equation is the same for all oscillators (in each ear). 
                # Note it's the same as for a single vodscillator, just with the interaural coupling!
            universal = z[k]*(1j*s.vod_L.omegas[k] + s.vod_L.epsilons[k] 
                              - (s.vod_L.alphas[k] + s.vod_L.betas[k]*1j)*((np.abs(z[k]))**2))
            # papilla coupling
            + s.P_CCC(P_l - z[k])
            # noise
            + s.vod_L.xi_glob(t) + s.vod_L.xi_loc[k](t) - s.xi(t)
            
            # HAIR BUNDLE NN COUPLING
            # if we're at an endpoint, we only have one oscillator to couple with
            if k == 0:
                ddt[k] = universal + s.vod_L.cccs[k]*(z[k+1] - z[k]) 
            elif k == s.vod_L.num_osc - 1:
                ddt[k] = universal + s.vod_L.cccs[k]*(z[k-1] - z[k])
            # but if we're in the middle of the chain, we couple with the oscillator on either side
            else:
                ddt[k] = universal + s.vod_L.cccs[k]*((z[k+1] - z[k]) + (z[k-1] - z[k])) 
                
        # RIGHT EAR
        for k in range(s.vod_L.num_osc, s.total_num_osc):
            # define an index "l" to go along the larger arrays
            l = k - s.vod_L.num_osc
            
            # The "universal" part of the equation is the same for all oscillators (in each ear). 
            universal = iac_right + (1j*s.vod_R.omegas[l] + s.vod_R.epsilons[l])*z[l] + s.vod_R.xi_glob(t) + s.vod_R.xi_loc[l](t) - (s.vod_R.alphas[l] + s.vod_R.betas[l]*1j)*((np.abs(z[l]))**2)*z[l]
            
            # Define the original Vodscillator inter-ear coupling
            
            # if we're at an endpoint, we only have one oscillator to couple with
            if k == 0:
                ddt[k] = universal + s.vod_R.cccs[l]*(z[l+1] - z[l])
            elif k == s.vod_R.num_osc - 1:
                ddt[k] = universal + s.vod_R.cccs[l]*(z[l-1] - z[l])
            # but if we're in the middle of the chain, we couple with the oscillator on either side
            else:
                ddt[k] = universal + s.vod_R.cccs[l]*((z[l+1] - z[l]) + (z[l-1] - z[l]))

        # P_l
        ddt[-3] = (s.K_T / s.M_T * (X_l - P_l) + s.K_C / s.M_T * (Z_c - P_l))*1j + P_l.imag
        # P_r
        ddt[-2] = (s.K_C / s.M_T * (Z_c - P_r) + s.K_T / s.M_T * (X_r - P_r))*1j + P_r.imag
        # Z_c
        ddt[-1] = (s.K_C / s.M_C * (P_l - Z_c) + s.K_C / s.M_C * (P_r - Z_c))*1j + Z_c.imag
                
        return ddt



    def save(s, filename = None):
        """ Saves your Twin Vodscillators in a .pkl file

        Parameters
        ------------
            filename: string, Optional
                Don't include the ".pkl" at the end of the string!
                If no filename is provided, it will just use the "name" given to your Twin Vodscillators
            
        """
        
        if filename:
            f = filename + ".pkl"
        else:
            f = s.name + ".pkl"

        with open(f, 'wb') as outp:  # Overwrites any existing file with this filename!.
            pickle.dump(s, outp, pickle.HIGHEST_PROTOCOL)

    def __str__(s):
        return f"A Twin Vodscillator named {s.name} with {s.vod_L.num_osc} oscillators on the left and {s.vod_R.num_osc} on the right!"
            
        

            
