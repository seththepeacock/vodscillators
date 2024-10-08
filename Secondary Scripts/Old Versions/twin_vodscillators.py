import numpy as np
import pickle
from vodscillator import *
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.fft import rfft, rfftfreq

class TwinVodscillators:
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
        s.iaccc = [["iaccc"]] # interaural complex coupling constant
        # check that our time parameters are the same for both or we might have some issues!
        vod_params = np.empty(2, 7)
        i = 0
        for v in [s.vl, s.vr]:
            vod_params[i, :] = [v.sample_rate, v.ti, v.tf, v.t_transient, v.t_win, v.num_wins, v.tpoints]
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
        s.ICs = np.concatenate(s.vl.ICs, s.vr.ICs)

        # Numerically integrate our ODE from ti to tf with sample rate 1/h
        sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y
        # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.

        # lets add a dimension at the beginning to track if its the left or right ear
            # in case each has different # oscillators, we'll use the max of the two
        max_num_osc = max(s.vl.num_osc, s.vr.num_osc)
        s.sol = np.zeros((2, max_num_osc, len(s.tpoints)), dtype=complex)
        s.sol[0, :, :] = sol[0:s.vl.num_osc, :]
        s.sol[1, :, :] = sol[s.vl.num_osc:(s.vl.num_osc+s.vr.num_osc), :]
        # so s.sol[1, 2, 1104] is the value of the solution for the 3rd oscillator in the right ear at the 1105th time point.

        # Now get the summed response of all the oscillators (SOO = Summed Over Oscillators)
        s.left_SOO_sol = np.sum(s.sol[0, :, :], 1)
        s.right_SOO_sol = np.sum(s.sol[1, :, :], 2)

    def ODE(s, t, z):
        # This function will only be called by the ODE solver

        # Mark the current point in time to track progress
        print(f"Time = {int(t)}/{int(s.tf)}")

        # First make an array to represent the current (complex) derivative of each oscillator
        ddt = np.zeros(s.total_num_osc, dtype=complex)


        # (We are adapting equation (11) in Vilfan & Duke 2008)
        # Define the interaural coupling expressions: iaccc (=interaural complex coupling constant) * summed response of the opposite ear
        interaural_left = s.iaccc * (np.sum(z[s.vl.num_osc:s.vr.num_osc]))
        interaural_right = s.iaccc * (np.sum(z[0:s.vr.num_osc]))

        # Now define our derivatives!
        # First, do the left ear. 
        for k in range(s.vl.num_osc):
            # The "universal" part of the equation is the same for all oscillators (in each ear). 
            # Note it's the same as for a single vodscillator, just with the interaural coupling!
            universal = interaural_left + (1j*s.vl.omegas[k] + s.vl.epsilon)*z[k] + ...
            s.vl.xi_glob(t) + s.vl.xi_loc[k](t) - (s.vl.alpha + s.vl.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]
            
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
        for k in np.arange(s.vl.num_osc, s.total_num_osc):
            # The "universal" part of the equation is the same for all oscillators (in each ear). 
            # Note it's the same as for a single vodscillator, just with the interaural coupling!
            universal = interaural_right + (1j*s.vr.omegas[k] + s.vr.epsilon)*z[k] + ...
            s.vr.xi_glob(t) + s.vr.xi_loc[k](t) - (s.vr.alpha + s.vr.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]
            
            # Coupling within each ear
            # if we're at an endpoint, we only have one oscillator to couple with
            if k == 0:
                ddt[k] = universal + s.vr.ccc*(z[k+1] - z[k])
            elif k == s.vr.num_osc - 1:
                ddt[k] = universal + s.vr.ccc*(z[k-1] - z[k])
            # but if we're in the middle of the chain, we couple with the oscillator on either side
            else:
                ddt[k] = universal + s.vr.ccc*((z[k+1] - z[k]) + (z[k-1] - z[k]))
                
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
        s.fft_freq = rfftfreq(s.n_win, 1/s.sample_rate)
        s.num_freq_points = len(s.fft_freq)
        
        # compute the (r)fft for all oscillators individually and store them in "every_fft"
            # note we are taking the r(eal)fft since (presumably) we don't lose much information by only considering the real part (position) of the oscillators  
        s.every_fft = np.zeros((2, s.total_num_osc, s.num_wins, s.num_freq_points), dtype=complex) # every_fft[l/r ear, osc index, which ss win, fft output]

        # we'll get the ss solutions:
        s.ss_sol = s.sol[:, :, s.n_transient:]
        
        for win in range(s.num_wins):
            # get the start and stop indices for this window
            n_start = win * s.n_win
            n_stop = (win + 1) * s.n_win
            # first, we'll calculate the ffts for the left side:
            for osc in range(s.vl.num_osc):
                # calculate fft
                s.every_fft[0, osc, win, :] = rfft((s.ss_sol[0, osc, n_start:n_stop]).real)
            # then we'll do the right side
            for osc in range(s.vr.num_osc):
                # calculate fft
                s.every_fft[1, osc, win, :] = rfft((s.ss_sol[1, osc, n_start:n_stop]).real)

        # finally, we'll add them all together to get the fft of the summed response (sum of fft's = fft of sum)
        s.left_SOO_fft = np.sum(s.every_fft[0], 0)
        s.right_SOO_fft = np.sum(s.every_fft[1], 0)

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
            
        

            

        