import numpy as np
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from matplotlib.cm import ScalarMappable

# Load the dataset
folders = os.listdir("data/radar-motor")
range_doppler_features = np.load("data/npz_files/radar-cfar-motor.npz", allow_pickle=True)

x_train, y_train = range_doppler_features['out_x'], range_doppler_features['out_y']

print(x_train.shape)
# Get initial class distribution
support_before, count_before = np.unique(y_train, return_counts=True)

# Reshape x_train for SMOTE
x_train = x_train.reshape(278, 16 * 128)

# Apply SMOTE
sm = SMOTE(random_state=2)
x_smote, y_smote = sm.fit_resample(x_train, y_train)

# Reshape x_smote back to original image dimensions
x_smote = x_smote.reshape(y_smote.shape[0], 16, 128)

print(x_smote.shape)

np.savez("data/npz_files/radar-balanced-cfar-motor.npz", out_x=x_smote, out_y=y_smote)
