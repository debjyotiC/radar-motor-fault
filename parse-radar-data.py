import pandas as pd
import numpy as np
from os import listdir
from os.path import isdir, join
from config_parser import parseConfigFile

dataset_path = 'data/radar-motor'
config_file_path = "data/config_files/motor-range-doppler.cfg"

configParameters = parseConfigFile(config_file_path, Rx_Ant=4, Tx_Ant=4)
configParameters["numDopplerBins"] = 16
configParameters["numRangeBins"] = 128

print(configParameters)

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    target_files = listdir(join(dataset_path, target))
    filenames.extend(target_files)
    y.extend([index] * len(target_files))


def apply_2d_cfar(signal, guard_band_width, kernel_size, threshold_factor):
    num_rows, num_cols = signal.shape
    threshold_signal = np.zeros((num_rows, num_cols))
    guard_band_area = (2 * guard_band_width + 1) ** 2 - (2 * kernel_size + 1) ** 2

    for i in range(guard_band_width, num_rows - guard_band_width):
        for j in range(guard_band_width, num_cols - guard_band_width):
            noise_level = (np.sum(
                signal[i - guard_band_width:i + guard_band_width + 1, j - guard_band_width:j + guard_band_width + 1]) -
                           np.sum(signal[i - kernel_size:i + kernel_size + 1,
                                  j - kernel_size:j + kernel_size + 1])) / guard_band_area
            threshold = threshold_factor * noise_level
            if signal[i, j] > threshold:
                threshold_signal[i, j] = 1
    return threshold_signal


def calc_range_doppler(data_frame, packet_id, config):
    payload = data_frame[packet_id].to_numpy()
    payload = 20 * np.log10(payload)
    rangeDoppler = np.reshape(payload, (config["numDopplerBins"], config["numRangeBins"]), 'F')
    rangeDoppler = np.roll(rangeDoppler, shift=len(rangeDoppler) // 2, axis=0)
    return rangeDoppler


out_x_range_doppler = []
out_x_range_doppler_cfar = []
out_y_range_doppler = []

for folder_idx, target in enumerate(all_targets):
    all_files = join(dataset_path, target)
    for file_name in listdir(all_files):
        full_path = join(all_files, file_name)
        print(full_path, folder_idx)

        df_data = pd.read_csv(full_path)

        for col in df_data.columns:
            data = calc_range_doppler(df_data, col, configParameters)
            cfar_data = apply_2d_cfar(data, guard_band_width=2, kernel_size=1, threshold_factor=1)

            out_x_range_doppler.append(data)
            out_x_range_doppler_cfar.append(cfar_data)
            out_y_range_doppler.append(folder_idx + 1)

data_range_x = np.array(out_x_range_doppler)
data_range_cfar_x = np.array(out_x_range_doppler_cfar)
data_range_y = np.array(out_y_range_doppler)

np.savez('data/npz_files/radar-motor.npz', out_x=data_range_x, out_y=data_range_y)
np.savez('data/npz_files/radar-cfar-motor.npz', out_x=data_range_cfar_x, out_y=data_range_y)
