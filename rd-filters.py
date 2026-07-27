import numpy as np
import cv2
from os import listdir
from config_parser import parseConfigFile
import matplotlib.pyplot as plt

# --- Load configuration & data ---
configParameters = parseConfigFile("data/config_files/motor-range-doppler.cfg", Rx_Ant=4, Tx_Ant=4)
data = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)
class_labels = listdir("data/radar-motor")

motor_data, motor_label = data['out_x'], data['out_y']

rangeArray = np.linspace(-1, 1, 16)
dopplerArray = np.linspace(0, 2, 128)

label_pos = 3

motor_data = motor_data[motor_label == label_pos]
motor_label = motor_label[motor_label == label_pos]

pos = 1  # Frame index to display

# --- Image Processing Pipeline ---
# 1. Grayscale normalization and upsampling
frame_norm = cv2.normalize(motor_data[pos], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
frame_uint8 = frame_norm.astype(np.uint8)
gray_img = cv2.resize(frame_uint8, (128, 128), interpolation=cv2.INTER_CUBIC)

# 2. Binary Thresholding (Otsu's Binarization)
_, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Sobel Edge Detection
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobel_x, sobel_y)
edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))  # Normalize to 0-255

# --- Side-by-Side Plotting ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle(f"Frame no. {pos} - Label: {class_labels[motor_label[pos] - 1]}", fontsize=14, fontweight='bold')

# 1. Original Raw Radar Matrix
axes[0].contourf(motor_data[pos], levels=100, cmap='seismic')
axes[0].set_title("Original (Raw Matrix)")
axes[0].set_xlabel("Doppler bins")
axes[0].set_ylabel("Range bins")

# 2. Grayscale Image
axes[1].contourf(gray_img, cmap='gray')
axes[1].set_title("Grayscale")
axes[1].set_xlabel("Doppler bins")
axes[1].set_ylabel("Range bins")

# 3. Binary Image
axes[2].contourf(binary_img, cmap='gray')
axes[2].set_title("Binary Image (Otsu)")
axes[2].set_xlabel("Doppler bins")
axes[2].set_ylabel("Range bins")

# 4. Sobel Edges
axes[3].contourf(edges_sobel, cmap='gray')
axes[3].set_title("Sobel Edges")
axes[3].set_xlabel("Doppler bins")
axes[3].set_ylabel("Range bins")

plt.tight_layout()
plt.savefig(f"images/{class_labels[motor_label[pos] - 1]}.jpg", dpi=600)
plt.show()