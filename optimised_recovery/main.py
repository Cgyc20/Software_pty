import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2
import torch
from imageprocessortorch import ImageProcessorTorch
from grad_des import grad_des_SGD

# Load the input image
image_path = "/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/conv_matrix/moto.png"
X = imread(image_path)
X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

# Parameters for gradient descent
psize = 3
pscaling = 1
sigma = 1
tv_factor = 0.1
iterations = 100
learning_rate = 0.01

p = np.array([1,1,1,1,1,1,1,1,1])
Probe = p.reshape((3,3))
# Create an instance of ImageProcessor
Model = ImageProcessorTorch(Probe)

# Input the image to ImageProcessor
Model.input_matrix(X)
blur = Model.create_blur()

print(type(blur))


# Create an instance of grad_des_SGD
optimizer = grad_des_SGD(Model,X,blur, tv_factor, randomised=True)

# Run SGD optimization
optimized_image = optimizer.run_SGD(iterations, learning_rate)

# Plot convergence
optimizer.plot_convergence()

# Plot final images
optimizer.plot_final()
