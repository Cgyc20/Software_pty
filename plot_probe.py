import numpy as np
import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
from pty import pty_torch
import torch 
import cv2

image_path = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/images/IMG_2587.png'

torch.set_grad_enabled(True) 
X = imread(image_path)
X_gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY) 
X = torch.tensor(X_gray, dtype=torch.float32, requires_grad=True)

# Instantiate pty_torch object with the original image X_gray

Probe_size = 25
scaling = 1
sigma = 10
conv_torch = pty_torch(X, Probe_size, scaling,sigma)

probe = conv_torch.Probe.numpy()

plt.figure()
plt.imshow(probe)
plt.colorbar()
plt.show()

