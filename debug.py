#%%
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
#%%
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def plot_spectrum(data):
    fft_results = np.fft.rfft(data)
    plt.plot(np.abs(fft_results))
    plt.show()
#%%
# Example usage:
fs = 1000.0  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector

# Create a sample EMG signal (a sum of two sine waves)
emg_signal = 0.7 * np.sin(2 * np.pi * 20 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)

# Bandpass filter parameters
lowcut = 40  # Lower cutoff frequency
highcut = 100  # Upper cutoff frequency

# Apply bandpass filter
filtered_emg = butter_bandpass_filter(emg_signal, lowcut, highcut, fs)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(t, emg_signal, label='Original EMG Signal')
plt.plot(t, filtered_emg, label='Filtered EMG Signal (Bandpass)')
plt.title('Original and Filtered EMG Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# %%
plot_spectrum(emg_signal)
# %%
plot_spectrum(filtered_emg)
# %%
