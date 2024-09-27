import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from config_parser import parseConfigFile

# Dataset paths
dataset_path = 'data/radar-motor'
config_file_path = "data/config_files/motor-range-doppler.cfg"

# Parse config file to get parameters
configParameters = parseConfigFile(config_file_path, Rx_Ant=4, Tx_Ant=4)

# Create range and doppler arrays based on the config
rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                           configParameters["dopplerResolutionMps"])

# Get folder names
folders = os.listdir(dataset_path)

# Load range-doppler features
range_doppler_features = np.load("data/npz_files/radar-balanced-motor.npz", allow_pickle=True)

# Extract x_train and y_train
x_train, y_train = range_doppler_features['out_x'], range_doppler_features['out_y']

# Get unique classes and their counts
classes, count = np.unique(y_train, return_counts=True)

sns.barplot(x=folders, y=count)
plt.show()

# # Initialize a count variable outside the loop
# count = 0
#
# def ca_cfar(matrix, guard_cells, training_cells, threshold_factor):
#     """
#     Apply CA-CFAR detection on a 2D range-Doppler matrix.
#
#     Parameters:
#     - matrix: 2D numpy array of the range-Doppler matrix
#     - guard_cells: Number of guard cells around the cell under test (CUT) [guard_cells_row, guard_cells_col]
#     - training_cells: Number of training cells around the guard cells [training_cells_row, training_cells_col]
#     - threshold_factor: Multiplicative factor for threshold calculation
#
#     Returns:
#     - detection_map: 2D numpy array with the same shape as matrix, where 1 indicates detection and 0 otherwise
#     """
#
#     num_rows, num_cols = matrix.shape
#     detection_map = np.zeros_like(matrix)
#
#     guard_rows, guard_cols = guard_cells
#     train_rows, train_cols = training_cells
#
#     # Iterate over each cell in the matrix, excluding the edges where CFAR can't be applied
#     for row in range(train_rows + guard_rows, num_rows - (train_rows + guard_rows)):
#         for col in range(train_cols + guard_cols, num_cols - (train_cols + guard_cols)):
#             # Define the neighborhood of the cell under test (CUT)
#             row_start = row - (train_rows + guard_rows)
#             row_end = row + (train_rows + guard_rows) + 1
#             col_start = col - (train_cols + guard_cols)
#             col_end = col + (train_cols + guard_cols) + 1
#
#             # Extract the training cells region excluding the guard cells and CUT
#             training_region = matrix[row_start:row_end, col_start:col_end]
#             cut_region = matrix[row - guard_rows:row + guard_rows + 1, col - guard_cols:col + guard_cols + 1]
#
#             # Mask out the guard cells and the CUT itself from the training region
#             mask = np.ones_like(training_region)
#             mask[train_rows:-train_rows, train_cols:-train_cols] = 0
#             masked_training_region = training_region * mask
#
#             # Compute the average of the training cells
#             training_cells_values = masked_training_region[np.nonzero(masked_training_region)]
#             noise_estimate = np.mean(training_cells_values)
#
#             # Calculate the threshold
#             threshold = noise_estimate * threshold_factor
#
#             # Compare the CUT value with the threshold
#             if matrix[row, col] > threshold:
#                 detection_map[row, col] = 1
#
#     return detection_map
#
# # Parameters
# guard_cells = [1, 2]  # Guard cells in both row and column directions
# training_cells = [3, 5]  # Training cells in both row and column directions
# threshold_factor = 1.0  # Example threshold factor
#
#
# # Loop through frames and labels to plot and save them
# for frame, label in zip(x_train, y_train):
#     plt.clf()
#     # plt.title(folders[label-1])
#     plt.xlabel("Range (m)")
#     plt.ylabel("Doppler velocity (m/s)")
#
#     # frame = ca_cfar(frame, guard_cells, training_cells, threshold_factor)
#
#     # Plot range-Doppler map with corrected extent
#     # plt.imshow(frame, extent=[rangeArray.min(), rangeArray.max(), dopplerArray.min(), dopplerArray.max()],
#     #            aspect='auto')
#
#     plt.contourf(frame)
#     # Increment count and save the plot
#     count += 1
#     plt.tight_layout()
#     # plt.savefig(f"images/{count}-range-doppler-ca-cfar-plot-{folders[label-1]}.png")
#
#     # Display the plot with a short pause
#     plt.pause(0.1)
