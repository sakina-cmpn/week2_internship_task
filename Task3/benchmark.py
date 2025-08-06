# benchmark.py

import time
import numpy as np
import pandas as pd
from timeseries_utils import rolling_mean_numpy, rolling_var_numpy, ewma_pandas, fft_bandpass

# Simulate time-series data
N = 1_000_000
np.random.seed(42)
data = np.random.randn(N)
df = pd.DataFrame({'value': data})
sampling_rate = 100  # Hz

results = []

# Rolling mean (NumPy)
start = time.time()
_ = rolling_mean_numpy(data, 100)
end = time.time()
results.append(["rolling_mean_numpy", end - start])

# Rolling mean (pandas)
start = time.time()
_ = df['value'].rolling(100).mean()
end = time.time()
results.append(["rolling_mean_pandas", end - start])

# Rolling var (NumPy)
start = time.time()
_ = rolling_var_numpy(data, 100)
end = time.time()
results.append(["rolling_var_numpy", end - start])

# Rolling var (pandas)
start = time.time()
_ = df['value'].rolling(100).var()
end = time.time()
results.append(["rolling_var_pandas", end - start])

# EWMA (pandas)
start = time.time()
_ = ewma_pandas(df['value'])
end = time.time()
results.append(["ewma_pandas", end - start])

# FFT filter
start = time.time()
_ = fft_bandpass(data, low_freq=1, high_freq=10, sampling_rate=sampling_rate)
end = time.time()
results.append(["fft_bandpass", end - start])

# Save benchmark results
benchmark_df = pd.DataFrame(results, columns=["Method", "Execution_Time_Seconds"])
benchmark_df.to_csv("benchmark_results.csv", index=False)
print(benchmark_df)
