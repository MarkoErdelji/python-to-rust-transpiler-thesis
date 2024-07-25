import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
data = pd.read_csv('epoch_times.csv')

def moving_average(series, window_size):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

window_size = 50


downsample_factor = 50

downsampled_data = data.iloc[::downsample_factor].reset_index(drop=True)

downsampled_data = downsampled_data[['Epoch', 'Time (ms)']]

downsampled_data.to_csv('smoothed_downsampled_data.csv', index=False)

print(f"Smoothed and downsampled data saved to 'smoothed_downsampled_data.csv' with window size {window_size} and downsample factor {downsample_factor}.")
