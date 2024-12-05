import numpy as np
from os import listdir
from config_parser import parseConfigFile
import matplotlib.pyplot as plt

configParameters = parseConfigFile("data/config_files/motor-range-doppler.cfg", Rx_Ant=4, Tx_Ant=4)

print(configParameters)

data = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)
class_labels = listdir("data/radar-motor")

motor_data, motor_label = data['out_x'], data['out_y']

rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]


def min_max_norm(mat):
    min_val = mat.min()
    max_val = mat.max()
    normalized_matrix = 2 * (mat - min_val) / (max_val - min_val) - 1
    return normalized_matrix


for count, frame in enumerate(motor_data):
    plt.clf()
    plt.title(f"Frame no. {count} has label {class_labels[motor_label[count] - 1]}")
    plt.contourf(min_max_norm(frame))
    plt.pause(0.5)
