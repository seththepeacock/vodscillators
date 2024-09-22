import numpy as np
import pickle
import random
from vodscillator import *
import matplotlib.pyplot as plt








# def do_fft(s):
#       """ Returns four arrays:
#       1. every_fft[oscillator index, ss win index, output]
#       2. SOO_fft[output]
#       3. AOI_fft[oscillator index, output]
#       4. SOO_AOI_fft[output]

#       AOI = Averaged Over Wins (for noise)

#       """
#       # first, we get frequency axis: the # of frequencies the fft checks depends on the # signal points we give it (n_win), 
#       # and sample spacing (h) tells it what these frequencies correspond to in terms of real time 
#       s.fft_freq = rfftfreq(s.n_win, 1/s.sample_rate)
#       s.num_freq_points = len(s.fft_freq)
      
#       # compute the (r)fft for all oscillators individually and store them in "every_fft"
#       # note we are taking the r(eal)fft since (presumably) we don't lose much information by only considering the real part (position) of the oscillators  
#       every_fft = np.zeros((2, s.total_num_osc, s.num_wins, s.num_freq_points), dtype=complex) # every_fft[l/r ear, osc index, which ss win, fft output]
#       T1_fft = np.zeros((s.num_wins, s.num_freq_points), dtype=complex)
#       T2_fft = np.zeros((s.num_wins, s.num_freq_points), dtype=complex)
      
#       # we'll get the ss solutions:
#       ss_sol = s.osc_sol[:, :, s.n_transient:]
#       T1_ss_sol = s.osc_sol[s.n_transient:]
#       T2_ss_sol = s.osc_sol[s.n_transient:]
      
#       for win in range(s.num_wins):
#       # get the start and stop indices for this window
#       n_start = win * s.n_win
#       n_stop = (win + 1) * s.n_win
#       # first, we'll calculate the ffts for the left side:
#       for osc in range(s.vl.num_osc):
#             # calculate fft
#             every_fft[0, osc, win, :] = rfft((ss_sol[0, osc, n_start:n_stop]).real)
#       # then we'll do the right side
#       for osc in range(s.vr.num_osc):
#             # calculate fft
#             every_fft[1, osc, win, :] = rfft((ss_sol[1, osc, n_start:n_stop]).real)
#       # then we'll get T1 and T2 ffts
#       T1_fft[win, :] = rfft(T1_ss_sol[n_start:n_stop].real)
#       T2_fft[win, :] = rfft(T2_ss_sol[n_start:n_stop].real)
      

#       # finally, we'll add them all together to get the fft of the summed response (sum of fft's = fft of sum)
#       s.left_SOO_fft = np.sum(every_fft[0], 0)
#       s.right_SOO_fft = np.sum(every_fft[1], 0)










# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
# a = a[1:-1, 1:-1]
# print(a[-1])

# First, generate time points with a little padding at the endpoints to smooth the interpolation
# s.padded_t = np.arange(s.ti, s.tf + 2*s.h, s.h)


# import json
# with open('output.txt', 'w') as filehandle:
#     json.dump(v.t.tolist(), filehandle)

# plt.plot(v.t, v.summed_sol.real, v.t, v.summed_sol.imag)
# plt.savefig("summed_response")
# plt.show()

# if not (None):
#     print("ey")



      # we're going to (temporarily) exclude the peak around 0 freq here!
      # min_freq_pt = int(0.1 * s.h)

      #s.avg_cluster_freqs[osc] = np.mean(s.AOI_fft_amps[osc][min_freq_pt:] / (2*np.pi))


#trying to troubleshoot by matching up with beth's code
# wf = v.summed_sol.real[v.n_transient:]


# specsum = rfft(wf[0:-1])/(len(wf[0:-1])/2) # getting the summed spectrum
# specsumdB = 20*np.log10(np.abs(specsum)) # take the summed spectrum in dB
# freqsum = rfftfreq(len(wf[0:-1]), v.h)  # freq of sum of all the oscillators 


# plt.figure(3)
# plt.title("Spectrum plot of sum of z")
# plt.plot(freqsum, specsumdB, '.-', label="Not averaged for noise")
# plt.xlim(0, 6.5)
# plt.xlabel('Frequency [kHz]')
# plt.ylabel('Magnitude [dB]')
# plt.legend()
# plt.grid()
# plt.show()

# clusters = np.empty(shape=(4, 4), dtype=list)
# clusters.fill([])
# clusters[3, 3].append(0)
# clusters[3, 3].append(2)
# clusters[3, 3].append(4)
# print(clusters)

# filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\"
# filename = "V&D fig 2A + APC.pkl"
# with open(filepath + filename, 'rb') as picklefile:
#     v = pickle.load(picklefile)
#     # this "assert" statement will let VSCode know that this is a Vodscillator, so it will display its documentation for you!
#     assert isinstance(v, Vodscillator)

#     print(v.apc)

# # nf=1000000
# # plt.figure(1)
# # plt.plot(vod.fft_freq, np.log10(vod.SOO_fft[0]/nf))
# # plt.xlim(0, 10)
# # plt.ylim(-5, 5)
# # nf=64
# # plt.figure(2)
# # plt.plot(vod.fft_freq, np.log10(vod.SOO_fft[0]/nf))
# # plt.xlim(0, 10)

# # plt.ylim(-5, 5)
# # plt.show()


# wf = np.linspace(0, 4066, 4067)
# print(wf)
# print(len(wf))
# n_shift = 1
# n_win = 1
# win_start_indices = np.arange(0, len(wf)-n_win+n_shift, n_shift)
# print(win_start_indices)
# num_wins = len(win_start_indices)
# windowed_wf = np.zeros((num_wins, n_win))
# for k in range(num_wins):
#       win_start = win_start_indices[k]
#       win_end = win_start + n_win
#       # grab the (real part of the) waveform in this window
#       windowed_wf[k, :] = wf[win_start:win_end]
#       # note this grabs the wf at indices win_start, win_start+1, ..., win_end-1
#       # if there are 4 samples and t_win=t_shift=1 and SR=1, then n_win=2, n_shift=1, and the first window will be samples 0 and 1, the next 1 and 2...
# print(windowed_wf)

# wf = np.linspace(0, 4066, 4067)
# t_win=1
# sample_rate=128
# t_shift=1
# # calculate the number of samples in the window
# # + 1 is because if you have SR=2 and you want a two second window, this will take 5 samples!
# n_win = t_win*sample_rate

# # and the number of samples to shift
# # no + 1 here; if you want to shift it over one second and SR=2, that will be two samples
# n_shift = t_shift*sample_rate

# # get sample_spacing
# sample_spacing = 1/sample_rate

# # if there are 4 samples and t_win=t_shift=1 and SR=1, then n_win=2, n_shift=1, and this will return [0, 1, 2] which is what we want
# # (2 is the index of the second to last sample, which is the start of the final window!)
# win_start_indices = np.arange(0, (len(wf)-n_win)+n_shift, n_shift)
# # the + n_shift is so that we actually include the endpoint since np.arange doesn't by default!

# # if number of windows is passed in, we make sure it's less than the length of win_starts
# # if no num_wins is passed in, we'll just use the max number of windows
# num_wins = len(win_start_indices)
# # print(win_start_indices)
# # print(num_wins)
# # print(n_win)
# # print(n_shift)
# # build windowed waveform array
# windowed_wf = np.zeros((num_wins, n_win))

# for k in range(num_wins):
#       win_start = win_start_indices[k]
#       win_end = win_start + n_win
#       # grab the (real part of the) waveform in this window
#       print(k)
#       windowed_wf[k, :] = wf[win_start:win_end].real
#       # note this grabs the wf at indices win_start, win_start+1, ..., win_end-1
#       # if there are 4 samples and t_win=t_shift=1 and SR=1, then n_win=2, n_shift=1 and
#       # Thus the first window will be samples 0 and 1, the next 1 and 2...


# def OLD_get_windowed_fft(wf, sample_rate, t_win, t_shift=None, num_wins=None):
#   """ Gets the windowed fft of the given waveform with given window size

#   Parameters
#   ------------
#       wf: array
#         waveform input array
#       sample_rate: int
#         defaults to 44100 
#       t_win: float
#         length (in time) of each window
#       num_wins: int, Optional
#         If this isn't passed, then just get the maximum number of windows of the given size

#   """
#   # if you didn't pass in t_shift we'll assume you want no overlap - each new window starts at the end of the last!
#   if t_shift is None:
#     t_shift=t_win

#   # get sample_spacing
#   sample_spacing = 1/sample_rate

#   # calculate num_win_pts
#   n_win = sample_rate * t_win + 1

#     # calculate number of windows (unless it's passed in)
#   if num_wins is None:
#     num_wins = int(len(wf) / n_win)

#   # initialize matrix which will hold the windowed waveform
#   windowed_wf = np.zeros((num_wins, n_win))
#   for win in range(num_wins):
#       win_start = win*n_win
#       win_end = (win+1)*n_win
#       # grab the (real part of the) waveform in this window
#       windowed_wf[win, :] = wf[win_start:win_end].real

#   # Now we do the ffts!

#   # get frequency axis 
#   freq_ax = rfftfreq(n_win, sample_spacing)
#   num_freq_pts = len(freq_ax)
#   # get fft of each window
#   windowed_fft = np.zeros((num_wins, num_freq_pts), dtype=complex)
#   for win in range(num_wins):
#     windowed_fft[win, :] = rfft(windowed_wf[win, :])
  
#   return freq_ax, windowed_fft


#  + 1 # num of time points corresponding to t_win 
      # NOTE the + 1 is a recent addition! if sample_rate=1, we need 2 samples to describe a 1 second
# scope = 2
# f = [5, 6, 7, 8, 9]
# k = 2
# start = k - scope
# end = k + scope + 1
# print(f[start:end])

# t_win = 10
# sr = 100
# n_win = sr * t_win
# print(rfftfreq(n_win, 1/sr))


#   if fcut:
#       fcut_index = np.where(freq_ax > 200)[0][0]
#       wfft2 = np.zeros((num_wins, num_freq_pts - fcut_index), dtype=complex)
#       freq_ax2 = np.zeros(num_freq_pts - fcut_index + 1)
#       for k in range(num_wins):
#         wfft2[k,:] = wfft[k, fcut_index:]
#         freq_ax2 = freq_ax[fcut_index:]


"NEVERMIND LOL TOO COMPLICATED"
              # get an array of unwrapped phases where first index is the modulo(bin_shift) group
              # unwrapped_phases = np.empty((bin_shift, np.size(phases, 0), int(np.size(phases, 1) / bin_shift)))
              # for i in range(bin_shift):
              #   # unwrap it w.r.t. "neighboring" frequency bins (axis 1)
              #   print(i)
              #   unwrapped_phases[i] = np.unwrap(phases[:, i::bin_shift], axis=1)"



"Specially unwrap freqs for get_coherence"

# # pass this in:
# special_unwrap_freq = None

# # optionally give special unwrapping treatment to a particular frequency bin 
# # so that the phase diffs between this and #(bin_shift) bins over are between -pi and pi
# if special_unwrap_freq is not None:
# special_freq_bin = int(special_unwrap_freq * t_win)
# k = int(np.mod(special_freq_bin, bin_shift))
# j = int(special_freq_bin / bin_shift)
# special_phases = np.unwrap(phases[:, k::bin_shift], 1)[:, j:j+2]  
# # calc phase diffs
# for win in range(num_wins):
#       for freq_bin in range(num_freq_pts - bin_shift):
#       phase_diffs[win, freq_bin] = phases[win, freq_bin + bin_shift] - phases[win, freq_bin]
#       if freq_bin == special_freq_bin:
#       phase_diffs[win, freq_bin] = special_phases[win, 1] - special_phases[win, 0]



"Prev_freq"
# elif ref_type == "prev_freq":
# # unwrap it w.r.t. neighboring frequency bins
# phases=np.unwrap(phases, axis=1)
# # initialize array for phase diffs; - 1 is because we won't be able to get it for the final freq 
# phase_diffs = np.zeros((num_wins, num_freq_pts - 1))
# # we'll also need to take the first bin off the freq_ax
# freq_ax = freq_ax[1:]

# # calc phase diffs
# for win in range(num_wins):
# for freq_bin in range(1, num_freq_pts):
#       # so the first entry is in phase_diffs[win, 0] and corresponds to the entry for phases[win, 1] which makes sense bc our first bin on freq_ax is the one that was originally at index 1
#       phase_diffs[win, freq_bin - 1] = phases[win, freq_bin] - phases[win, freq_bin - 1]
# coherence = get_vector_strength(phase_diffs)
# # put the phase diffs in the output dict
# d["phase_diffs"] = phase_diffs

list = np.array([236, 573, 135, 47, 7, 42])
print(np.mean(list**2))
print(np.mean(list)**2)