from vodscillators import *
import matplotlib.pyplot as plt
import timeit
from numpy.fft import rfft, rfftfreq
from statistics import mean

# CREATE AND SAVE A VODSCILLATOR

start = timeit.default_timer() # starts timer that tells you code runtime

p = {
#General Initializing Params
"name" : "V+D fig 2",
"num_osc" : 80, # number of oscillators in chain[default = 100 or 150], 80 in paper

#set_freq
"freq_dist" : "linear", #linear or exp
"roughness_amp" : 0,
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
"num_intervals" : 5, # [default for no noise is 1; when we have noise we average over multiple intervals, default = 30]
"sample_rate" : 128, #[default = 128]

#set_ODE"
"epsilon" : 4.0, # [default = 1.0] --> control parameter
"d_R" : 16, # [default = 0.15] --> real part of coupling coefficient
"d_I" : 0, # [default = -1.0] --> imaginary part of coupling coefficient
"B" : 1.0, # [default = 1.0] --> amount of cubic nonlinearity
}

v = Vodscillator(**p)
v.set_freq(**p)
v.set_ICs(**p)
v.gen_noise(**p)
v.solve_ODE(**p)
v.sum_solution()
v.get_fft()
v.save()

stop = timeit.default_timer() # ends timer
print('Total time:', stop - start, "seconds, or", (stop-start)/60, "minutes") 
# prints the total time the code took to run



