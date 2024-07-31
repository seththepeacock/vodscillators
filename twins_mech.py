from vodscillator import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from plots import *
from vlodder import *
import scipy.io


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

    k_T = 1 #coupling with tympanum
    N = s.vl.num_osc 
    m_0 = 1 #mass of single vodscillator
    iaccc = k_T / N / m_0 #iac coupling constant
    k_C = 1
    m_C = 1
    m_T = 1
    T_1 = 0
    T_2 = 0
    X_C = 0


    # This function will only be called by the ODE solver
    # Mark the current point in time to track progress
    print(f"Time = {int(t)}/{int(s.tf)}")
    # First make an array to represent the current (complex) derivative of each oscillator
    ddt = np.zeros(s.total_num_osc+3, dtype=complex) #last 3 are T_1, T_2 and X_C
    # (We are adapting equation (11) in Vilfan & Duke 2008)
    # Define the interaural coupling expressions: iaccc (=interaural complex coupling constant) * summed response of the opposite ear
    X_1 = np.sum(z[s.vl.num_osc:s.vr.num_osc])
    X_2 = np.sum(z[0:s.vr.num_osc])
    interaural_left = -1j*iaccc * (X_1 - T_1)
    interaural_right = 1j*iaccc * (T_2 - X_2)
    # Now define our derivatives!
    # First, do the left ear. 
    for k in range(s.vl.num_osc):
        # The "universal" part of the equation is the same for all oscillators (in each ear). 
        # Note it's the same as for a single vodscillator, just with the interaural coupling!
        universal = interaural_left + (1j*s.vl.omegas[k] + s.vl.epsilon)*z[k] + s.vl.xi_glob(t) + s.vl.xi_loc[k](t) - (s.vl.alpha + s.vl.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]
        
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
        universal = interaural_right + (1j*s.vr.omegas[k] + s.vr.epsilon)*z[k] + s.vr.xi_glob(t) + s.vr.xi_loc[k](t) - (s.vr.alpha + s.vr.betas[k]*1j)*((np.abs(z[k]))**2)*z[k]
        
        # Coupling within each ear
        # if we're at an endpoint, we only have one oscillator to couple with
        if k == 0:
            ddt[k] = universal + s.vr.ccc*(z[k+1] - z[k])
        elif k == s.vr.num_osc - 1:
            ddt[k] = universal + s.vr.ccc*(z[k-1] - z[k])
        # but if we're in the middle of the chain, we couple with the oscillator on either side
        else:
            ddt[k] = universal + s.vr.ccc*((z[k+1] - z[k]) + (z[k-1] - z[k]))

    ddt[-3] = k_T / m_T * (X_1 - T_1) + k_C / m_T*  (X_C - T_1)
    ddt[-2] = k_C / m_T * (X_C - T_2) + k_T / m_T * (X_C - T_1)
    ddt[-1] = k_T / m_T * (T_1 - X_1) + k_C / m_C * (T_2 - X_2)
            
    return ddt