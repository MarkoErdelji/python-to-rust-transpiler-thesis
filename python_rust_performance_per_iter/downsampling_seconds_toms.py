import pandas as pd
import numpy as np

data = pd.read_csv('svm_epoch_times.csv')

def moving_average(series, window_size):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

window_size = 50

data['Time (seconds)'] = moving_average(data['Time (seconds)'], window_size)

data['Time (milliseconds)'] = data['Time (seconds)'] * 1e3

downsample_factor = 50

downsampled_data = data.iloc[::downsample_factor].reset_index(drop=True)

downsampled_data.to_csv('smoothed_downsampled_data.csv', index=False)

print(f"Smoothed and downsampled data saved to 'smoothed_downsampled_data.csv' with window size {window_size} and downsample factor {downsample_factor}.")
