import numpy as np
from os import listdir
import matplotlib.pyplot as plt

data = np.load("data/npz_files/radar-cfar-motor.npz", allow_pickle=True)
class_labels = listdir("data/radar-motor")

motor_data, motor_label = data['out_x'], data['out_y']

for count, frame in enumerate(motor_data):
    plt.clf()
    # frame = apply_2d_cfar(frame, guard_band_width=2, kernel_size=5, threshold_factor=1)
    plt.title(f"Frame no. {count} has label {class_labels[motor_label[count] - 1]}")
    plt.contourf(frame)
    plt.pause(1)
