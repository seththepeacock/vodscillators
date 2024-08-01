import numpy as np
import pickle
from vodscillator import *
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq

class Twins:
    """
    Vod-structions
    1. Create Two Vodscillators
    2. Initialize frequency distribution, ICs, and (optional) non-isochronicity with "initialize(**p)"
    3. Generate Noise with "gen_noise(**p)"
    4. Create your TwinVodscillator

    """
    def __init__(s, vl=Vodscillator, vr=Vodscillator, **p):
        # s = self (refers to the object itself)
        # **p unpacks the dictionary of parameters (p) we pass into the initializer
        s.vl = vl #left Vodscillator
        s.vr = vr #right Vodscillator
        s.name = p["name"] # name your vodscillator! (also will be used as a filename)
        s.glob_glob_noise_amp = p["glob_glob_noise_amp"] # amplitude of (uniformly generated) noise
        
        # check that our time parameters are the same for both or we might have some issues!
        vod_params = np.empty((2, 6))
        i = 0
        for v in [s.vl, s.vr]:
            vod_params[i, :] = np.array([v.sample_rate, v.ti, v.tf, v.t_transient, v.t_win, v.num_wins])

            i = 1
        # if they agree, set all of the TV's parameters to them
        if all(vod_params[0, :] == vod_params[1, :]):
            print("Vods agree!")
            s.sample_rate = vl.sample_rate
            s.ti = vl.ti
            s.tf = vl.tf
            s.t_transient = vl.t_transient
            s.t_ss = vl.t_win
            s.n_transient = vl.n_transient
            s.n_win = vl.n_win
            s.num_wins = vl.num_wins
            s.tpoints = vl.tpoints
        else:
            print("Parameter mismatch!")

        # generate noise
        global_global_noise = np.random.uniform(-s.glob_glob_noise_amp, s.glob_glob_noise_amp, len(s.tpoints)) 
        # then interpolate between (using a cubic spline) for ODE solving adaptive step purposes
        s.xi_glob_glob = CubicSpline(s.tpoints, global_global_noise)
        # define some useful parameters
        s.total_num_osc = s.vl.num_osc + s.vr.num_osc
      

    def solve_ODE(s):
        # NOTE WE ALWAYS PUT LEFT BEFORE RIGHT
        
        # create ICs by concatenating left and right IC arrays
        s.ICs = np.concatenate((s.vl.ICs, s.vr.ICs, np.array([0, 0, 0])))
        
        # Numerically integrate our ODE (making sure that it gives us the value at our tpoints)
        sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y
        
        # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
        # lets add a dimension at the beginning to track if its the left or right ear
            # in case each has different # oscillators, we'll use the max of the two
        max_num_osc = max(s.vl.num_osc, s.vr.num_osc)
        s.osc_sol = np.zeros((2, max_num_osc, len(s.tpoints)), dtype=complex)
        s.osc_sol[0, :, :] = sol[0:s.vl.num_osc, :]
        s.osc_sol[1, :, :] = sol[s.vl.num_osc:(s.total_num_osc), :]
        
        # so s.sol[1, 2, 1104] is the value of the solution for the 3rd oscillator in the right ear at the 1105th time point.
        # Now get the summed response of all the oscillators (SOO = Summed Over Oscillators)
        s.SOOL_sol = np.sum(s.osc_sol[0, :, :], 0)
        s.SOOR_sol = np.sum(s.osc_sol[1, :, :], 0)
        s.T_l_sol = sol[-3]
        s.T_r_sol = sol[-2]
        s.X_c_sol = sol[-1]

    def ODE(s, t, z):

        K_T = 1 # tympanum spring constant
        K_C = 1 # cavity spring constant
        M_0 = 1 # mass of single vodscillator hair bundle
        M_C = 1 # mass of the air in the IAC
        M_T = 1 # mass of the tympanum

        # This function will only be called by the ODE solver
        # Mark the current point in time to track progress
        print(f"Time = {int(t)}/{int(s.tf)}")
        # First make an array to represent the current (complex) derivative of each oscillator
        ddt = np.zeros(s.total_num_osc+3, dtype=complex) #last 3 are T_1, T_2 and X_C
        # (We are adapting equation (11) in Vilfan & Duke 2008)

        # get variables:
        
        # summed vodscillators
        X_l = np.sum(z[s.vl.num_osc:s.vr.num_osc])
        X_r = np.sum(z[0:s.vr.num_osc])
        # tympani
        T_l = z[-3]
        T_r = z[-2]
        # air cavity
        X_c = z[-1]
        iac_left = K_T / s.vl.num_osc / M_0 * (X_l - T_l) * 1j
        iac_right = K_T / s.vr.num_osc / M_0 * (T_r - X_r) * 1j
        
        # Now define our derivatives!
        
        # First, do the left ear. 
        for k in range(s.vl.num_osc):
            # The "universal" part of the equation is the same for all oscillators (in each ear). 
                # Note it's the same as for a single vodscillator, just with the interaural coupling!
            universal = iac_left + (1j*s.vl.omegas[k] + s.vl.epsilon)*z[k] + s.vl.xi_glob(t) + s.vl.xi_loc[k](t) - (s.vl.alpha + s.vl.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]
            
            # Coupling within each ear
            # if we're at an endpoint, we only have one oscillator to couple with
            if k == 0:
                ddt[k] = universal + s.vl.ccc*(z[k+1] - z[k]) 
            elif k == s.vl.num_osc - 1:
                ddt[k] = universal + s.vl.ccc*(z[k-1] - z[k])
            # but if we're in the middle of the chain, we couple with the oscillator on either side
            else:
                ddt[k] = universal + s.vl.ccc*((z[k+1] - z[k]) + (z[k-1] - z[k])) 
        # Now, do the right ear.
        for k in range(s.vl.num_osc, s.total_num_osc):

            l = k - s.vl.num_osc
            # The "universal" part of the equation is the same for all oscillators (in each ear). 
                # Again, it's the same as for a single vodscillator, just with the interaural coupling!
            universal = iac_right + (1j*s.vr.omegas[l] + s.vr.epsilon)*z[l] + s.vr.xi_glob(t) + s.vr.xi_loc[l](t) - (s.vr.alpha + s.vr.betas[l]*1j)*((np.abs(z[l]))**2)*z[l]
            
            # Coupling within each ear
            
            # if we're at an endpoint, we only have one oscillator to couple with
            if k == 0:
                ddt[k] = universal + s.vr.ccc*(z[l+1] - z[l])
            elif k == s.vr.num_osc - 1:
                ddt[k] = universal + s.vr.ccc*(z[l-1] - z[l])
            # but if we're in the middle of the chain, we couple with the oscillator on either side
            else:
                ddt[k] = universal + s.vr.ccc*((z[l+1] - z[l]) + (z[l-1] - z[l]))

        # set derivatives for the new state variables T1, T2, X_C
        ddt[-3] = (K_T / M_T * (X_l - T_l) + K_C / M_T * (X_c - T_l))*1j + T_l.imag
        ddt[-2] = (K_C / M_T * (X_c - T_r) + K_T / M_T * (X_r - T_r))*1j + T_r.imag
        ddt[-1] = (K_C / M_C * (T_l - X_c) + K_C / M_C * (T_r - X_c))*1j + X_c.imag
                
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
        return f"A Twin Vodscillator named {s.name} with {s.vl.num_osc} oscillators on the left and {s.vr.num_osc} on the right!"
            
        

            
