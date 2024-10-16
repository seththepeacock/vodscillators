import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.io
from vodscillator import *
from plots import *
from vlodder import *
from twinvods import *
from scipy.signal import find_peaks

def smooth_data(x, y, window_size=5):
    """
    Smooth the data using a simple moving average.
    
    Parameters:
    - y: array-like, y values to smooth
    - window_size: int, size of the moving average window
    
    Returns:
    - y: array, smoothed y values
    """
    trim_amount = (window_size - 1) // 2
    x = x[trim_amount : len(x) - trim_amount]
    y = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    return x, y

def nth_highest_local_maximum(x, y, n, min_distance=10, min_prominence=0.000001):
    """
    Find the nth highest local maximum of the given curve with smoothing and filtering.
    
    Parameters:
    - x: array-like, x values of the curve
    - y: array-like, y values of the curve
    - n: int, the rank of the local maximum to find (1 for the highest, 2 for the second highest, etc.)
    - min_distance: int, minimum distance between peaks
    - min_prominence: float, minimum prominence of peaks
    
    Returns:
    - nth_max_x: float, x value of the nth highest local maximum
    - nth_max_y: float, y value of the nth highest local maximum
    """
    
    # Find local maxima using scipy's find_peaks with distance and prominence
    peaks, properties = find_peaks(y, distance=min_distance, prominence=min_prominence)

    # Extract the y values of the local maxima
    peak_values = y[peaks]
    
    # Combine peak indices and values, and sort them by values
    sorted_peaks = sorted(zip(peaks, peak_values), key=lambda x: x[1], reverse=True)

    # Check if there are enough peaks to return the nth highest
    if n > len(sorted_peaks):
        raise ValueError(f"Requested {n}th highest local maximum, but only {len(sorted_peaks)} local maxima found.")

    # Get the nth highest peak
    nth_peak = sorted_peaks[n] #note this starts with the "0th" biggest peak
        
    # Extract x and y values of the nth peak
    nth_max_x = x[nth_peak[0]]
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
sr = 44100

# set FT parameters
trunc = 1 
t_win = 1
# set parameters (in hertz)
freq_min = 1000
freq_max = 5000
min_peak_distance_hertz = 50 
smoothing_window_hertz = 11
# set lorentzian parameters
min_lorentzians = 9
max_lorentzians = 10
max_retries = 5   # Maximum number of retries for fitting


# get data / calculate some things
mat = scipy.io.loadmat(filepath + filename)
wf = np.squeeze(mat['wf'])
wf = wf[0:int(len(wf) / trunc)]
d = get_mags(wf, sr, t_win=t_win, dict=True)

x_data = d["freq_ax"]
y_data = np.log10(d["mags"])


min_i = np.where(x_data == freq_min)[0][0] 
max_i = np.where(x_data == freq_max)[0][0]
x_data = x_data[min_i:max_i]
y_data = y_data[min_i:max_i]
# convert from hertz to samples (each freq bin is 1/t_win wide, so for each hertz we need t_win # bins to fill it out)
min_distance = int(min_peak_distance_hertz * t_win)
smoothing_window = int(smoothing_window_hertz * t_win)


# get smoothed y_data
# x_smooth, y_smooth = smooth_data(x_data, y_data, smoothing_window)
x_smooth, y_smooth = x_data, y_data

# show us the smoothed data + picked out maxes
if 1==0:
    plt.plot(x_smooth, y_smooth)
    for i in range(max_lorentzians):
        x, y = nth_highest_local_maximum(x_smooth, y_smooth, i, min_distance=min_distance)
        plt.scatter(x, y)
        
    plt.show()


# Set up some things
best_aic = float('inf')
best_params = None
best_n = 0

# Attempt fitting for a varying number of Lorentzians
for n in range(min_lorentzians, max_lorentzians + 1):
    fitting_success = False
    attempt = 0
    
    while not fitting_success and attempt < max_retries:
        initial_guess = []
        
        for i in range(n):
            x0, y0 = nth_highest_local_maximum(x_smooth, y_smooth, i, min_distance=min_distance)
            gamma = np.random.uniform(0.1, 2.0)  # Random width
            
            # A is just the height of the peak y0, x0 is location of peak x0, and gamma is random
            initial_guess.extend([y0, x0, gamma])
            
            # on the first attempt, print out all the maxes we found
            if attempt == 0:
                print(f"Max #{i} is at {x0}")
        
        # Attempt to fit the combined Lorentzian model to the data
        try:
            params, covariance = curve_fit(combined_lorentzian, x_smooth, y_smooth, p0=initial_guess)
            fitting_success = True
            y_fit = combined_lorentzian(x_smooth, *params)
            print(f"Fitting succeeded for {n} Lorentzians with parameters: {params}")
            residuals = y_smooth - y_fit
            rss = np.sum(residuals**2)  # Residual sum of squares
            aic = 2 * len(params) + len(y_smooth) * np.log(rss / len(y_smooth))  # AIC calculation
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
        y_best_fit = combined_lorentzian(x_smooth, *best_params)
    else:
        print(f"Failed to fit {n} Lorentzians after {max_retries} attempts.")
        
# Print the best fit parameters
print("Best fit with {} Lorentzians:".format(best_n))
for i in range(best_n):
    print(f"Lorentzian {i + 1}: A={best_params[3 * i]:.3f}, x0={best_params[3 * i + 1]:.3f}, gamma={best_params[3 * i + 2]:.3f}")

# Generate fitted data for plotting
y_best_fit = combined_lorentzian(x_smooth, *best_params)





# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='blue', s=10)
plt.plot(x_smooth, y_best_fit, label='Best Fitted Combined Lorentzian', color='red')
plt.plot(x_smooth, y_smooth, label="Smoothed Data", color='green')
plt.title('Fitting Multiple Lorentzian Functions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
