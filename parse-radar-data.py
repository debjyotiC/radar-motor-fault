import random
import pandas as pd
import numpy as np
from os import listdir
from os.path import isdir, join

dataset_path = 'data/radar-motor'

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)

filenames = []
y = []

for index, target in enumerate(all_targets):
    target_files = listdir(join(dataset_path, target))
    filenames.extend(target_files)
    y.extend([index] * len(target_files))


def calc_range_doppler(data_frame, packet_id,):
    payload = data_frame[packet_id].to_numpy()
    payload = 20 * np.log10(payload)
    range_doppler = np.reshape(payload, (16, 128), 'F')
    range_doppler = np.roll(range_doppler, shift=len(range_doppler) // 2, axis=0)
    return range_doppler


out_x_range_doppler = []
out_y_range_doppler = []

for folder_idx, target in enumerate(all_targets):
    all_files = join(dataset_path, target)
    for file_name in listdir(all_files):
        full_path = join(all_files, file_name)
        print(full_path, folder_idx)

        df_data = pd.read_csv(full_path)

        for col in df_data.columns:
            data = calc_range_doppler(df_data, col)
            out_x_range_doppler.append(data)
            out_y_range_doppler.append(folder_idx + 1)

data_range_x = np.array(out_x_range_doppler)
data_range_y = np.array(out_y_range_doppler)

np.savez('data/npz_files/radar-motor.npz', out_x=data_range_x, out_y=data_range_y)
