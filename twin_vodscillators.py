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
        # check that our time parameters are the same for both or we might have some issues!
        vod_params = np.empty(2, 7)
        i = 0
        for v in [s.vl, s.vr]:
            vod_params[i, :] = [v.sample_rate, v.ti, v.tf, v.t_transient, v.t_ss, v.num_intervals, v.tpoints]
            i = 1
        # if they agree, set all of the TV's parameters to them
        if all(vod_params[0, :] == vod_params[1, :]):
            print("Vods agree!")
            s.sample_rate = vl.sample_rate
            s.ti = vl.ti
            s.tf = vl.tf
            s.t_transient = vl.t_transient
            s.t_ss = vl.t_ss
            s.n_transient = vl.n_transient
            s.n_ss = vl.n_ss
            s.num_intervals = vl.num_intervals
            s.tpoints = vl.tpoints
        else:
            print("Parameter mismatch!")

        # generate noise
        global_global_noise = np.random.uniform(-s.glob_glob_noise_amp, s.glob_glob_noise_amp, len(s.tpoints)) 
        # then interpolate between (using a cubic spline) for ODE solving adaptive step purposes
        s.xi_glob_glob = CubicSpline(s.tpoints, global_global_noise)

    def solve_ODE(s):
        # NOTE WE ALWAYS PUT LEFT BEFORE RIGHT

        # create ICs by concatenating left and right IC arrays
        s.ICs = np.concatenate(s.vl.ICs, s.vr.ICs)

        # Numerically integrate our ODE from ti to tf with sample rate 1/h
        sol = solve_ivp(s.ODE, [s.ti, s.tf], s.ICs, t_eval=s.tpoints).y
        # adding ".y" grabs the solutions - an array of arrays, where the first dimension is oscillator index.
        # so s.sol[2, 1104] is the value of the solution for the 3rd oscillator at the 1105th time point.

        # lets add a dimension at the beginning to track if its the left or right ear
        # in case each has different oscillators, we'll use the max of the two
        max_num_osc = max(s.vl.num_osc, s.vr.num_osc)
        s.sol = np.zeros((2, max_num_osc, s.tpoints), dtype=complex)
        s.sol[0, :, :] = sol[0:s.vl.num_osc, :]
        s.sol[1, :, :] = sol[s.vl.num_osc:(s.vl.num_osc+s.vr.num_osc), :]

        # Now get the summed response of all the oscillators (SOO = Summed Over Oscillators)
        s.left_SOO_sol = np.sum(s.sol[0, :, :], )
        s.right_SOO_sol = np.sum(s.sol)

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
            
        

            

        