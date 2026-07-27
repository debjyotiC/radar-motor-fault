import numpy as np
import cv2
from os import listdir
from config_parser import parseConfigFile
import matplotlib.pyplot as plt

# --- Load configuration & data ---
data = np.load("data/npz_files/radar-motor.npz", allow_pickle=True)
class_labels = listdir("data/radar-motor")

motor_data, motor_label = data['out_x'], data['out_y']

rangeArray = np.linspace(-1, 1, 128)
dopplerArray = np.linspace(0, 2, 128)

label_pos = 3
pos = 1

motor_data = motor_data[motor_label == label_pos][pos]
motor_label = motor_label[motor_label == label_pos][pos]


# --- Image Processing Pipeline ---
# 1. Upscale the RD image
upscale_img = cv2.resize(motor_data, (128, 128), interpolation=cv2.INTER_CUBIC)

# 2. Grayscale normalisation
gray_img = cv2.normalize(upscale_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
gray_img = gray_img.astype(np.uint8)

# 3. Binary Thresholding (Otsu's Binarization)
_, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Sobel Edge Detection
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobel_x, sobel_y)
edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))  # Normalize to 0-255

# --- Side-by-Side Plotting ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
# fig.suptitle(f"Frame no. {pos} - Label: {class_labels[motor_label - 1]}", fontsize=14, fontweight='bold')

# 1. Original Raw Radar Matrix
axes[0].contourf(dopplerArray, rangeArray, upscale_img, levels=100, cmap='seismic')
axes[0].set_title("Original (Raw Matrix)")
axes[0].set_ylabel("Doppler velocity (m/s^2)")
axes[0].set_xlabel("Range (m)")

# 2. Grayscale Image
axes[1].contourf(dopplerArray, rangeArray, gray_img, cmap='gray')
axes[1].set_title("Grayscale")
axes[1].set_ylabel("Doppler velocity (m/s^2)")
axes[1].set_xlabel("Range (m)")

# 3. Binary Image
axes[2].contourf(dopplerArray, rangeArray, binary_img, cmap='gray')
axes[2].set_title("Binary Image (Otsu)")
axes[2].set_ylabel("Doppler velocity (m/s^2)")
axes[2].set_xlabel("Range (m)")

# 4. Sobel Edges
axes[3].contourf(dopplerArray, rangeArray, edges_sobel, cmap='gray')
axes[3].set_title("Sobel Edges")
axes[3].set_ylabel("Doppler velocity (m/s^2)")
axes[3].set_xlabel("Range (m)")

plt.tight_layout()
plt.savefig(f"images/{class_labels[motor_label - 1]}.jpg", dpi=600)
plt.show()
