import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.io
from vodscillator import *
from plots import *
from vlodder import *
from twinvods import *
from scipy.signal import find_peaks

def smooth_data(y, window_size=5):
    """
    Smooth the data using a simple moving average.
    
    Parameters:
    - y: array-like, y values to smooth
    - window_size: int, size of the moving average window
    
    Returns:
    - smoothed_y: array, smoothed y values
    """
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

def nth_highest_local_maximum(x, y, n, smoothing_window=5, min_distance=10, min_prominence=0.000001):
    """
    Find the nth highest local maximum of the given curve with smoothing and filtering.
    
    Parameters:
    - x: array-like, x values of the curve
    - y: array-like, y values of the curve
    - n: int, the rank of the local maximum to find (1 for the highest, 2 for the second highest, etc.)
    - smoothing_window: int, size of the moving average window for smoothing
    - min_distance: int, minimum distance between peaks
    - min_prominence: float, minimum prominence of peaks
    
    Returns:
    - nth_max_x: float, x value of the nth highest local maximum
    - nth_max_y: float, y value of the nth highest local maximum
    """
    # Smooth the y values
    smoothed_y = smooth_data(y, window_size=smoothing_window)
    
    # Adjust x values to match the length of smoothed y
    adjusted_x = x[(smoothing_window - 1) // 2: -(smoothing_window // 2)]
    
    # Find local maxima using scipy's find_peaks with distance and prominence
    peaks, properties = find_peaks(smoothed_y, distance=min_distance, prominence=min_prominence)

    # Extract the y values of the local maxima
    peak_values = smoothed_y[peaks]
    
    # Combine peak indices and values, and sort them by values
    sorted_peaks = sorted(zip(peaks, peak_values), key=lambda x: x[1], reverse=True)
    
    # print(sorted_peaks)

    # Check if there are enough peaks to return the nth highest
    if n > len(sorted_peaks):
        raise ValueError(f"Requested {n}th highest local maximum, but only {len(sorted_peaks)} local maxima found.")

    # Get the nth highest peak
    nth_peak = sorted_peaks[n] #note this starts with the "0th" biggest peak
        
    # Extract x and y values of the nth peak
    nth_max_x = adjusted_x[nth_peak[0]]
    nth_max_y = nth_peak[1]

    return nth_max_x, nth_max_y

# Define the Lorentzian function
def lorentzian(x, A, x0, gamma):
    return A / ((x - x0)**2 + gamma**2)

# Define the combined Lorentzian function
def combined_lorentzian(x, *params):
    n = len(params) // 3  # Number of Lorentzian components
    y = np.zeros_like(x)
    for i in range(n):
        A = params[3 * i]
        x0 = params[3 * i + 1]
        gamma = params[3 * i + 2]
        y += lorentzian(x, A, x0, gamma)
    return y


# LOAD DATA
filepath = "C:\\Users\\Owner\\OneDrive\\Documents\\GitHub\\vodscillators\\SOAE Data\\"
# filename = 'TH14RearwaveformSOAE.mat'
filename = 'ACsb24rearSOAEwfA1'
mat = scipy.io.loadmat(filepath + filename)
wf = np.squeeze(mat['wf'])
sr = 44100

# parameters for fourier transform
trunc_fraction = 1 # divide wf length by this number
t_win = 10 # window size for FT
freq_min = 1000 # minimum frequency to keep
freq_max = 5000 # max frequency to keep
# parameters for identifying peaks
min_peak_distance_hertz = 50 # in hertz
smoothing_window_hertz = 10 # we'll do some smoothing to make it easier to pick out the peaks (smoothing window size in hertz)
# parameters for the
min_lorentzians = 7 # minimum number of lorentz functions to fit
max_lorentzians = 10 # maximum number of lorentz functions to fit
rounds_per_num_lorentzians = 2 # how many (ideally successful) rounds we do for each number of lorentzians
max_retries = 3   # Maximum number of retries per round (when it doesn't converge) before we just move on


# define some things
wf = wf[0:int(len(wf) / trunc_fraction)]
d = get_mags(wf, sr, t_win=t_win, dict=True)

x_data = d["freq_ax"]
y_data = d["mags"]

min_i = np.where(x_data == freq_min)[0][0] 
max_i = np.where(x_data == freq_max)[0][0]
x_data = x_data[min_i:max_i]
y_data = y_data[min_i:max_i]

best_aic = float('inf')
best_params = None
best_n = 0


# convert from hertz to samples (each freq bin is 1/t_win wide, so for each hertz we need t_win # bins to fill it out)
min_peak_sample_distance = int(min_peak_distance_hertz * t_win)
smoothing_Window = int(smoothing_window_hertz * t_win)


# Attempt fitting for a varying number of Lorentzians
for n in range(min_lorentzians, max_lorentzians + 1):
    print(f"{n} Lorentzians")
    # Try each each one until you've successfully (or failed enough retried) found tries_per_num_lorentzians for each # of lorentzians 
    for k in range(1, rounds_per_num_lorentzians + 1):
        print(f"Number {k}")
        fitting_success = False
        attempt = 0
        # for each round, if we fail to converge we'll try again until we hit max_retries
        while not fitting_success and attempt < max_retries:
            initial_guess = []
            
            for i in range(n):
                # A = np.random.uniform(0.1, 2.0)  # Random amplitude
                # x0 = np.random.uniform(0, 100)  # Random center
                x0, y0 = nth_highest_local_maximum(x_data, y_data, i, min_distance=min_peak_sample_distance)
                # print(f"Max #{i} is at {x0}")
                gamma = np.random.uniform(0.1, 5.0)  # Random width
                initial_guess.extend([y0, x0, gamma])
            
            # Attempt to fit the combined Lorentzian model to the data
            try:
                params, covariance = curve_fit(combined_lorentzian, x_data, y_data, p0=initial_guess)
                fitting_success = True
                y_fit = combined_lorentzian(x_data, *params)
                print(f"Fitting succeeded for {n} Lorentzians with parameters: {params}")
                residuals = y_data - y_fit
                rss = np.sum(residuals**2)  # Residual sum of squares
                aic = 2 * len(params) + len(y_data) * np.log(rss / len(y_data))  # AIC calculation
                # Update best AIC and parameters
                if aic < best_aic:
                    best_aic = aic
                    best_params = params
                    best_n = n
            except RuntimeError as e:
                print(f"Fitting failed for {n} Lorentzians on attempt {attempt + 1}: {e}")
                attempt += 1
                continue
            except Exception as e:
                print(f"An unexpected error occurred for {n} Lorentzians on attempt {attempt + 1}: {e}")
                attempt += 1
                continue

        if fitting_success:
            y_best_fit = combined_lorentzian(x_data, *best_params)
        else:
            print(f"Failed to fit {n} Lorentzians after {max_retries} attempts.")
        
# Print the best fit parameters
print("Best fit with {} Lorentzians:".format(best_n))
for i in range(best_n):
    print(f"Lorentzian {i + 1}: A={best_params[3 * i]:.3f}, x0={best_params[3 * i + 1]:.3f}, gamma={best_params[3 * i + 2]:.3f}")

# Generate fitted data for plotting
y_best_fit = combined_lorentzian(x_data, *best_params)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='blue', s=10)
plt.plot(x_data, y_best_fit, label='Best Fitted Combined Lorentzian', color='red')
plt.title('Fitting Multiple Lorentzian Functions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
