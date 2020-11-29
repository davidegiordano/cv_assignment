#!/usr/bin/env python
# coding: utf-8

# # Assignment
# ## Computer Vision and Image Processing
# ### Prof: Luigi Di Stefano, luigi.distefano@unibo.it
# ### Tutor: Pierluigi Zama Ramirez, pierluigi.zama@unibo.it - Riccardo Spezialetti, riccardo.spezialetti@unibo.it

# ## KITTI
# [KITTI](http://www.cvlibs.net/datasets/kitti/index.php) is a collection of 42,382 stereo sequences taken in urban environments from two video cameras and a LiDAR device mounted on the roof of a car. This dataset is widely used for benchmarking geometric understanding tasks such as depth, flow and pose estimation. 
# 
# <img src="kitti/000000_10.png" width="720">
# 
# ## Goal
# Given a Kitti image alongside the camera instrisics parameters, transform the image as it would have been acquired by your phone camera, i.e. by having your mobile phone mounted in the very exact position and pose as the Kitti camera.
# 
# ## Data
# **Images**: Images in the folder: _"kitti/"_
# 
# **Intrinsics Parameters of the KITTI Camera (Matrix A)**: 
# $$\begin{bmatrix} 707.0912 & 0.0 & 601.8873 \\ 0.0 & 707.0912 & 183.1104 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$

# In[24]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Find chessboard corners

def processImage(fn):
    print('processing {}'.format(fn))
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # Check image loaded correctly
    if img is None:
        print("Failed to load", fn)
        return None
    # Finding corners
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if found:
        # Refining corner position
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        # Visualize detected corners
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        plt.figure(figsize=(20,10))
        plt.imshow(vis)
        plt.show()
    else:
        print('chessboard not found')
        return None
    print('           %s... OK' % fn)
    return (corners.reshape(-1, 2), pattern_points)

dirname = "es0/"
img_names = [dirname + str(i) + ".png" for i in range(12)]

pattern_size = (8,5) # number of inner corner, (columns, rows) for OpenCV
square_size = 26.5 #mm

indices = np.indices(pattern_size, dtype=np.float32)
indices *= square_size
coords_3D = np.transpose(indices, [2, 1, 0])
coords_3D = coords_3D.reshape(-1, 2)
pattern_points = np.concatenate([coords_3D, np.zeros([coords_3D.shape[0], 1], dtype=np.float32)], axis=-1)

chessboards = [processImage(fn) for fn in img_names]

# Creating the lists of 2D and 3D points
obj_points = [] #3D points
img_points = [] #2D points

for (corners, pattern_points) in chessboards:
    img_points.append(corners)
    obj_points.append(pattern_points)


# Getting the width and height of the images
h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]

# Calibrating Camera
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())
print("Rotation vectors:", rvecs)
print("translation vectors", tvecs)


# In[25]:


# Write here your solution


img = cv2.imread('kitti/000000_10.png')
plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

kitti_camera_matrix = np.array([[707.0912, 0.0, 601.8873],
                         [0.0, 707.0912, 183.1104],
                         [0.0, 0.0, 1.0]])
#h, w = img.shape[:2]
print(kitti_camera_matrix)
#print(w, h)

# Finding the new optical camera matrix
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, 0.0, (w, h), 1, (w, h))

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
print(newcameramtx)

im_undistorted = cv2.undistort(img, kitti_camera_matrix, dist_coefs, None, newcameramtx)
# im_undistorted = cv2.undistort(img, camera_matrix, 0.0, None, newcameramtx)

# mx, my = cv2.initUndistortRectifyMap(camera_matrix, np.asarray([0]*4), None, newcameramtx, (w, h), 5) #calculate conversion from A to CAo
# im_undistorted = cv2.remap(img, mx, my, cv2.INTER_LINEAR) # convert to CAo
    
x, y, w_2, h_2 = roi
im_undistorted = im_undistorted[y:y+h_2, x:x+w_2]

# Plotting UNDISTORTED image
plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(im_undistorted, cv2.COLOR_BGR2RGB))
plt.show()

