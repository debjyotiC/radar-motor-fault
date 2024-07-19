import numpy as np
from os import listdir
import matplotlib.pyplot as plt

data = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)
class_labels = listdir("data/radar-motor")

motor_data, motor_label = data['out_x'], data['out_y']


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


for count, frame in enumerate(motor_data):
    plt.clf()
    frame = apply_2d_cfar(frame, guard_band_width=2, kernel_size=1, threshold_factor=1)
    plt.title(f"Frame no. {count} has label {class_labels[motor_label[count] - 1]}")
    plt.contourf(frame)
    plt.pause(1)
