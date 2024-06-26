from vodscillators import *
import matplotlib.pyplot as plt
import timeit
import pickle

# import json

# with open('output.txt', 'w') as filehandle:
#     json.dump(v.t.tolist(), filehandle)


# plt.plot(v.t, v.summed_sol.real, v.t, v.summed_sol.imag)
# plt.savefig("summed_response")
# plt.show()


with open("frank.pkl", 'rb') as picklefile:
    v = pickle.load(picklefile)

# Now we can use our vodscillator v with its solution pre-solved for!

# Generating V&D figs 2-5 style clustering graphs

charfreq = v.omegas / (2*np.pi)






