import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
from pty import pty_functions #This is the class I wrote, containing the convolution process
import cv2 


X = imread('/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/images/IMG_2587.png') #Import the image
#Convert X to a grayscale image 
X_gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY) #Convert to grayscale


P_size = 11 #The size of the probe
scaling = 1 #The scaling factor (Ie the relative radius of the probe)
method = 'Gaussian' #The values of the probe. 'Gaussian' will fill in the two dimensional normal distribution, 
#Constant will fill in with just ones. 
sigma = 2
Conv = pty_functions(X_gray, P_size, scaling, method,sigma, grayscale = False,strategy = 'extrapolate')

#Y = Conv.convolution_process()  # Call the convolution method on the instance. THis is the final image output.
b = Conv.sample_from_convolution(1) #Taking skipped values

Conv.plot(b)


probe = Conv.Probe
