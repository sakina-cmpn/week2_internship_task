# timeseries_utils.py

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Rolling mean using NumPy stride tricks
def rolling_mean_numpy(arr, window):
    if window > len(arr):
        raise ValueError("Window size larger than array.")
    windows = sliding_window_view(arr, window_shape=window)
    return windows.mean(axis=-1)

# Rolling variance using NumPy
def rolling_var_numpy(arr, window):
    if window > len(arr):
        raise ValueError("Window size larger than array.")
    windows = sliding_window_view(arr, window_shape=window)
    return windows.var(axis=-1)

# EWMA using pandas
def ewma_pandas(df, alpha=0.3):
    return df.ewm(alpha=alpha).mean()

# FFT-based band-pass filter
def fft_bandpass(signal, low_freq, high_freq, sampling_rate):
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    filtered_fft = fft_vals * mask
    filtered_signal = np.fft.ifft(filtered_fft).real
    return filtered_signal