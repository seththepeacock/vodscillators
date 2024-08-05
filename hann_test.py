import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

# Define the number of samples, frequency of the sine wave, and sampling rate
N = 1024
frequency = 5  # Frequency of the sine wave in Hz
sampling_rate = 100  # Sampling rate in Hz

# Create a time array and the sine wave
t = np.arange(N) / sampling_rate
sine_wave = np.sin(2 * np.pi * frequency * t)

# Apply the Hanning window to the sine wave
windowed_sine_wave = sine_wave * np.hanning(N)

# Compute the FFT of the windowed sine wave
fft_result_sine = fft(windowed_sine_wave)

# Calculate the phase of the FFT result
phase_sine = np.angle(fft_result_sine)

# Unwrap the phase to remove discontinuities
unwrapped_phase_sine = np.unwrap(phase_sine)


# Plot the unwrapped phase
plt.figure(figsize=(10, 4))
plt.plot(unwrapped_phase_sine)
plt.title("Unwrapped Phase of FFT of Hanning-Windowed Sine Wave")
plt.xlabel("Frequency Bin")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.show()
