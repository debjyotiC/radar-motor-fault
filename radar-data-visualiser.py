import numpy as np
import matplotlib.pyplot as plt

range_doppler_features = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)

x_train, y_train = range_doppler_features['out_x'], range_doppler_features['out_y']

print(x_train.shape)
print(y_train.shape)
