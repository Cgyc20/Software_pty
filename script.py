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



def main(Probe_size,scaling,sigma,skip_value,iter,tv_factor,lr,factor,bool):
    image_path = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/images/IMG_2587.png'

    torch.set_grad_enabled(True) 
    X = imread(image_path)
    X_gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY) 
    X = torch.tensor(X_gray, dtype=torch.float32, requires_grad=True)

    # Instantiate pty_torch object with the original image X_gray
    conv_torch = pty_torch(X, Probe_size, scaling,sigma)
    model = grad_des_SGD(X,Probe_size,scaling,sigma,skip_value,tv_factor,Randomised=bool)

    model.random_create(factor) #Note if we aren't using a random matrix then uncomment this line!




    blurred_image = model.B.detach().numpy()
    # fft_blurred_image = np.fft.fftshift(np.fft.fft2(blurred_image))
    # fft_abs_squared = np.abs(fft_blurred_image)**2
    # fft_abs_squared_log = np.log(fft_abs_squared + 1)  # Add 1 to avoid log(0)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(fft_abs_squared_log, cmap='gray')
    # plt.title('Log of Squared Modulus of Fourier Transform of Blurred Image')
    # plt.colorbar()
    # plt.show()

    model.run_SGD(iterations=iter,learning_rate=lr)
    model.plot_convergence()
    model.plot_final() 
    model.plot_final_two() 


if __name__ == '__main__':
    dict = {'Probe_size': 11,
            'scaling': 1,
            'sigma': 10,
            'skip_value': 5,
            'tv_factor':0.45,
            'iter': 5000,
            'lr': 1*1e-4,
            'factor': 0.2,
            'bool': False}
    for key, value in dict.items():
        print(f'{key}: {value}')
    main(**dict)
