import numpy as np
from os import listdir
import matplotlib.pyplot as plt

folders = listdir("data/radar-motor")
data = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)

motor_data, motor_label = data['out_x'], data['out_y']

labels, counts = np.unique(motor_label, return_counts=True)

print(folders)

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(folders, counts, width=0.4)

plt.xlabel("Labels")
plt.ylabel("No. of data points")
plt.show()

