from vodscillators import *
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft, rfftfreq
from statistics import mean

# CREATE AND SAVE A VODSCILLATOR

start = timeit.default_timer() # starts timer that tells you code run time

p = {
#General Initializing Params
"name" : "Frank",
"num_osc" : 100, # number of oscillators in chain[default = 100 or 150], 80 in paper

#set_freq
"freq_dist" : "exp", #linear or exp
"roughness_amp" : 1,
"omega_0" : 2*np.pi, # char frequency of lowest oscillator [default = 2*np.pi] 
"omega_N" : 5*np.pi, # char frequency of highest oscillator [default = 5*(2*np.pi)] 

#set_ICs
"IC_method" : "rand", #rand or const

#gen_noise
"loc_noise_amp" : 0.1, #amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
"glob_noise_amp" : 0.1, #amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
"ti" : 0, # start time; [default = 0]
"n_transient" : 35855, # the # of time points we give for transient behavior to settle down; around 30000 [default = 35855]
"n_ss" : 8192, # the # of time points we observe the steady state behavior for [default = 8192]
"num_runs" : 30, # [default for no noise is 1; when we have noise we average over multiple runs]
"sample_rate" : 128, #[default = 128]

#set_ODE
"epsilon" : 1.0, # [default = 1.0] --> control parameter
"d_R" : 0.15, # [default = 0.15] --> real part of coupling coefficient
"d_I" : -1.0, # [default = -1.0] --> imaginary part of coupling coefficient
"B" : 1.0, # [default = 1.0] --> amount of cubic nonlinearity
}

v = Vodscillator(**p)
v.set_freq(**p)
v.set_ICs(**p)
v.gen_noise(**p)
v.set_ODE(**p)
v.solve_ODE(**p)
v.save("frank.pkl")



stop = timeit.default_timer() # ends timer
print('Total time:', stop - start, "seconds, or", (stop-start)/60, "minutes") 
# prints the total time the code took to run









    





