import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

# Define the number of samples and create the Hanning window
N = 1024
hanning_window = np.hanning(N)

#sine = np.sin(np.linspace(0, np.pi * 20, 1024))


# Compute the FFT of the Hanning window
fft_result = fft(hanning_window)

# Calculate the magnitude of the FFT
magnitude = np.abs(fft_result)

# Plot the Hanning window and its FFT magnitude
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Plot the Hanning window
ax[0].plot(hanning_window)
ax[0].set_title("Hanning Window")
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Amplitude")

# Plot the magnitude of the FFT
ax[1].plot(magnitude)
ax[1].set_title("FFT Magnitude")
ax[1].set_xlabel("Frequency Bin")
ax[1].set_ylabel("Magnitude")
ax[1].set_yscale('log')
plt.tight_layout()
plt.show()
