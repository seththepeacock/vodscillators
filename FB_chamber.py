from vodscillator import *
import matplotlib.pyplot as plt
import timeit
from scipy.fft import rfft, rfftfreq
from statistics import mean

# CREATE AND SAVE A VODSCILLATOR

start = timeit.default_timer() # starts timer that tells you code runtime

p = {
#General Initializing Params
"name" : "F&B fig 2D iso",
"num_osc" : 50, # number of oscillators in chain[default = 100 or 150], 80 in paper

#set_freq
"freq_dist" : "exp", #linear or exp
"roughness_amp" : 0,
"omega_0" : 5, # char radial frequency of lowest oscillator [default = 2*np.pi] 
"omega_N" : 25, # char radial frequency of highest oscillator [default = 5*(2*np.pi)] 

#set_ICs
"IC_method" : "rand", #rand or const

#gen_noise
"loc_noise_amp" : 0.1, #amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
"glob_noise_amp" : 0.1, #amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
"ti" : 0, # start time; [default = 0]
"t_transient" : 280, # how long we give for transient behavior to settle down [default = 280 --> n.transient = 35840]
"t_ss" : 64, # length of an interval of ss observation [default = 64 --> n.transient = 8192]
"num_intervals" : 30, # [default for no noise is 1; when we have noise we average over multiple intervals, default = 30]
"sample_rate" : 128, #[default = 128]

#set_ODE"
"epsilon" : 1.0, # [default = 1.0] --> control parameter
"d_R" : 2.0, # [default = 0.15] --> real part of coupling coefficient
"d_I" : -2.0, # [default = -1.0] --> imaginary part of coupling coefficient
"alpha" : 1.0, # [default = 1.0] --> real coefficient for cubic nonlinearity
"beta_sigma" : 0.0 # [0 = isochronous as in V&D] --> std dev for imaginary coefficient for cubic nonlinearity (beta_j) which creates nonisochronicity
}

v = Vodscillator(**p)
v.initialize(**p)
v.gen_noise(**p)
v.solve_ODE()
v.do_fft()
v.save()

stop = timeit.default_timer() # ends timer
print('Total time:', stop - start, "seconds, or", (stop-start)/60, "minutes") 
# prints the total time the code took to run



