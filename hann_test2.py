import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

# Define the number of samples, frequency of the sine wave, and sampling rate
N = 1024
frequency = 5  # Frequency of the sine wave in Hz
sampling_rate = 100  # Sampling rate in Hz

# Create a time array and the sine wave
t = np.arange(N) / sampling_rate

# Apply the Hanning window to the sine wave
hann = np.hanning(N)

# Compute the FFT of the windowed sine wave
fft_result = fft(hann)

# Calculate the phase of the FFT result
phase = np.angle(fft_result)

# Unwrap the phase to remove discontinuities
unwrapped_phase = np.unwrap(phase)


# Plot the unwrapped phase
plt.figure(figsize=(10, 4))
plt.plot(unwrapped_phase, label="unwrapped")
plt.plot(phase, label="wrapped")
plt.title("(Unwrapped) Phase of FFT of Hanning-Window")
plt.xlabel("Frequency Bin")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.legend()
plt.show()
