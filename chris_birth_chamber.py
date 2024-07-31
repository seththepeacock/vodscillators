from vodscillator import *
import timeit
import pickle


def save_SOO_wf(vod=Vodscillator):
    filename = "wf - " + v.name + ".pkl"
    wf = vod.SOO_sol[vod.n_transient:]
    with open(filename, 'wb') as outp:  # Overwrites any existing file with this filename!.
        pickle.dump(wf, outp, pickle.HIGHEST_PROTOCOL)


# CREATE AND SAVE A VODSCILLATOR

start = timeit.default_timer() # starts timer that tells you code runtime

p = {
# General Initializing Params
"name" : "V&D fig 4, loc=0.1, glob=0, sr=128, rough=0.1",
"num_osc" : 80, # number of oscillators in chain[default = 100 or 150], 80 in paper

# initialize
"freq_dist" : "exp", #linear or exp
"roughness_amp" : 0.1,
"omega_0" : 2*np.pi, # char frequency of lowest oscillator [default = 2*np.pi] 
"omega_N" : 5*(2*np.pi), # char frequency of highest oscillator [default = 5*(2*np.pi)] 
"IC_method" : "rand", #rand or const

# gen_noise
"loc_noise_amp" : 0.1, #amplitude (sigma value) for local noise [0 --> off, default = 0.1-5]
"glob_noise_amp" : 0, #amplitude (sigma value) for global noise [0 --> off, default = 0.1-5]
"ti" : 0, # start time; [default = 0]
"t_transient" : 280, # how long we give for transient behavior to settle down [default = 280 --> n.transient = 35840]
"t_win" : 64, # length of a win of ss observation [default = 64 --> n.transient = 8192]
"num_wins" : 650, # [default for no noise is 1; when we have noise we average over multiple wins, default = 30]
"sample_rate" : 128, #[default = 128]

# solve_ODE
"epsilon" : 1.0, # [default = 1.0] --> control parameter
"d_R" : 0.15, # [default = 0.15] --> real part of coupling coefficient
"d_I" : -1.0, # [default = -1.0] --> imaginary part of coupling coefficient
"alpha" : 1.0, # [default = 1.0] --> real coefficient for cubic nonlinearity (in V&D, B = alpha + beta*i)
"beta_sigma" : 0.0 # [0 = isochronous as in V&D] --> std dev (normal dist w/ 0 mean) for beta_j... 
# imaginary coefficient for cubic nonlinearity (beta_j) which creates nonisochronicity
}

v = Vodscillator(**p)
v.initialize(**p)
v.gen_noise(**p)
v.solve_ODE()
v.save()


# v = Vodscillator(**p)
# v.initialize(**p)
# v.gen_noise(**p)
# v.solve_ODE()
# save_SOO_wf(v, "wf - " + v.name)

# print("Finished with WF 1")

# p["loc_noise_amp"] = 0.1
# v = Vodscillator(**p)
# v.initialize(**p)
# v.gen_noise(**p)
# v.solve_ODE()
# v.name = f"V&D fig 4, loc={v.loc_noise_amp}, glob={v.glob_noise_amp}, sr={v.sample_rate}"
# save_SOO_wf(v)

# print("Finished with WF 2")

# p["glob_noise_amp"] = 0.1
# v.initialize(**p)
# v.gen_noise(**p)
# v.solve_ODE()
# v.name = f"V&D fig 4, loc={v.loc_noise_amp}, glob={v.glob_noise_amp}, sr={v.sample_rate}"
# save_SOO_wf(v)

# print("Finished with WF 3")






