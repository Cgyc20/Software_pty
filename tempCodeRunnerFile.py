import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from pty import grad_des_SGD, pty_torch



def main(Probe_size,scaling,skip_value,iter,lr,factor,Randomised):
    image_path = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/images/IMG_2587.png'

    torch.set_grad_enabled(True) 
    X = imread(image_path)
    X_gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY) 
    X = torch.tensor(X_gray, dtype=torch.float32, requires_grad=True)

    # Instantiate pty_torch object with the original image X_gray
    conv_torch = pty_torch(X, Probe_size, scaling)
    model = grad_des_SGD(X,Probe_size,scaling,skip_value,Randomised = False)
    model.random_create(factor)
    model.run_SGD(iterations=iter,learning_rate=lr)
    model.plot_convergence()
    model.plot_final() 


if __name__ == '__main__':
    dict = {'Probe_size': 9,
            'scaling': 1,
            'skip_value': 1, 
            'iter': 200,
            'lr': 1e-4,
            'factor': 0.5,
            'Randomised': False}
    for key, value in dict.items():
        print(f'{key}: {value}')

    main(**dict)
