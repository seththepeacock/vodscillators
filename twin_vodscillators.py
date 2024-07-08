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

  def __init__(s, vl=Vodscillator, vr=Vodscillator, name=str):
    # s = self (refers to the object itself)
    # **p unpacks the dictionary of parameters (p) we pass into the initializer
    s.vl = vl #left Vodscillator
    s.vr = vr #right Vodscillator
    s.name = name # name your vodscillator! (also will be used as a filename)

    # check that our time parameters are the same for both or we might have some issues!
    vod_params = np.empty(2, 10)
    i = 0
    for v in [s.vl, s.vr]:
      vod_params[i, 0] = v.sample_rate
      vod_params[i, 1] = v.ti
      vod_params[i, 2] = v.tf
      vod_params[i, 3] = v.t_ss
      vod_params[i, 4] = v.num_intervals
    if all(vod_params[0, :] == vod_params[1, :]):
        print("Vods agree!")
    else:
       print("Parameter mismatch!")


  def gen_noise(s, twin_glob_noise_amp):


     

    