import numpy as np
from os import listdir
import matplotlib.pyplot as plt

data = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)
class_labels = listdir("data/radar-motor")

motor_data, motor_label = data['out_x'], data['out_y']

for count, frame in enumerate(motor_data):
    plt.clf()
    plt.title(f"Frame no. {count} has label {class_labels[motor_label[count]-1]}")
    plt.contourf(frame)
    plt.pause(0.5)
