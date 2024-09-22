from vodscillator import *
from twins import *
import timeit

# CREATE AND SAVE A TWIN VODSCILLATOR SYSTEM

start = timeit.default_timer() # starts timer that tells you code runtime

p = {
# General Initializing Params
"num_osc" : 80, # number of oscillators in chain[default = 100 or 150], 80 in paper
"name" : "V&D fig 5, loc=0.0785, glob=0, sr=128", # name your vodscillator!

# initialize()

"IC_method" : "rand", #rand or const
"freq_dist" : "exp", #linear or exp
"omega_0" : 2*np.pi*1000, # char frequency of lowest oscillator [default = 2*np.pi] 
"omega_N" : 5*(2*np.pi)*1000, # char frequency of highest oscillator [default = 5*(2*np.pi)] 

# if the np's (below) are nonzero, then these become the mean values rather than the parameters themselves
"epsilon" : 628, # [default = 628] --> control parameter
"alpha" : 6280, # [default = 6280] --> real coefficient for cubic nonlinearity (V&D's "B" is our alpha_k + beta_k*i)
"d_R" : 157, # [default = 157] --> real part of coupling coefficient
"d_I" : -754, # [default = -754] --> imaginary part of coupling coefficient


# these will be the *percent* of noise (np = noise percent) in each oscillators parameters... that is:
    # epsilons[i] = random number uniformly pulled from the range [epsilon +/- epsilon_np*epsilon]
"epsilon_np" : 0.20, # [default = 0.20]
"d_R_np" : 0.20, # [default = 0.20]
"d_I_np" : 0.20, # [default = 0.20]
"alpha_np" : 0, # [default = 0]
"omega_np" : 0.01, #[default = 0.01]

# nonisochronicity
"beta_sigma" : 0.0, # [0 = isochronous as in V&D] --> std dev (normal dist w/ 0 mean) for beta_k (V&D's "B" is our alpha_k + beta_k*i)

# gen_noise()
"loc_noise_amp" : 0.0785, #amplitude for local noise [0 --> off, default = 0.1-5] corresponds to D(tilda) in V&D
"glob_noise_amp" : 0, #amplitude for global noise [0 --> off, default = 0.1-5]
"ti" : 0, # start time; [default = 0]
"t_transient" : 280, # how long we give for transient behavior to settle down [default = 280 --> n.transient = 35840]
"t_win" : 64, # length of a win of ss observation [default = 64 --> n.transient = 8192]
"num_wins" : 30, # [default for no noise is 1; when we have noise we average over multiple wins, default = 30]
"sample_rate" : 128, #[default = 128]
}

vod = Vodscillator(**p)
vod.initialize(**p)
vod.gen_noise(**p)
vod.solve_ODE()
vod.save()


stop = timeit.default_timer() # ends timer
print('Total time:', stop - start, "seconds, or", (stop-start)/60, "minutes") 
# prints the total time the code took to run



