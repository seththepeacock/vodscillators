from vodscillator import *
from vlodder import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pickle
from plots import *

# calculates APC and then save to file
def apc_and_save(vod=Vodscillator, cluster_width=float, t_win=float, amp_weights=bool, num_wins=float, f_min=float, f_max=float, f_resolution=float):
    # calculate the apc, and it'll be (temporarily) saved to the vod object
    apc = get_apc(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)
    # pickle the apc into its own file
    with open(f"cluster_width={cluster_width}, num_wins={num_wins}, t_win={t_win}, amp_weights={amp_weights}.pkl", 'wb') as outp:  # Overwrites any existing file with this filename!.
        pickle.dump(apc, outp, pickle.HIGHEST_PROTOCOL)

# open up a vod
# filepath = "/home/deniz/Dropbox/vodscillators/"
# filename = "F&B fig 2D, noniso, loc=0.1, glob=0.1, sr1024, 15numint.pkl"

filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\Pickle Jar\\"
filename = "V&D fig 4, loc=0.1, glob=0, sr=128.pkl"
with open(filepath + filename, 'rb') as picklefile:
    vod = pickle.load(picklefile)
    # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
    assert isinstance(vod, Vodscillator)

# define your parameters
f_resolution=0.01
f_min=0
f_max=7
num_wins=100

# start with a pretty big cluster_width (THIS IS BETTER THAN 0.01 or 0.001)
cluster_width=0.1

t_win=1
amp_weights=True
# run fx to get apc and save to a lil pickle
apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

# change any parameters you want and rerun (note you can copy and paste the same list of args)
amp_weights=False
apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)


# a bunch more generating
if 1==1:
    t_win=8
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=32
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=1/2
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)


    t_win=1/8
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=1/32
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    # change cluster_width then run it all again
    cluster_width=0.05

    t_win=1
    amp_weights=True
    # run fx to get apc and save to a lil pickle
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    # change any parameters you want and rerun (note you can copy and paste the same list of args)
    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=8
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=32
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=1/8
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=1/32
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    # change cluster_width then run it all again
    cluster_width=0.2

    t_win=1
    amp_weights=True
    # run fx to get apc and save to a lil pickle
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    # change any parameters you want and rerun (note you can copy and paste the same list of args)
    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=8
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=32
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=1/8
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    t_win=1/32
    amp_weights=True
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)

    amp_weights=False
    apc_and_save(vod, cluster_width=cluster_width, t_win=t_win, amp_weights=amp_weights, num_wins=num_wins, f_min=f_min, f_max=f_max, f_resolution=f_resolution)