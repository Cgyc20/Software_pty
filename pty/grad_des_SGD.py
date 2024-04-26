import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
from .pty_torch import pty_torch
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm


# Load and preprocess the input image
class grad_des_SGD:
     
    def __init__(self,X,psize,pscaling,sigma,skip,tv_factor,Randomised = False):
        torch.set_grad_enabled(True) 
        self.psize = psize
        self.sigma = sigma
        self.skip = skip
        self.X = X
        self.pscaling = pscaling
        self.tv_factor = tv_factor
        self.conv_torch = pty_torch(X, psize, pscaling,sigma)
        # Get the sampled matrix B
        self.B = self.conv_torch.sample_from_convolution(skip) #Sampled matrix B   
        self.B_np = self.B.detach().numpy()

        if Randomised == True:
            self.Y_tensor = torch.randn(X.shape, dtype=torch.float32, requires_grad=True)
            print("Using entirely random Initial matrix")
        else:
            pass
        

    def random_create(self,factor):
        self.Y_tensor =torch.tensor(self.X + factor*torch.randn(self.X.shape, dtype=torch.float32),requires_grad=True)
        

    def __total_variation(self,image):
        # Compute finite differences along height and width dimensions
        dh = image[1:, :] - image[:-1, :]
        dw = image[:, 1:] - image[:, :-1]
        # Compute total variation
        tv = torch.sum(torch.abs(dh)) + torch.sum(torch.abs(dw))
        return tv

# Define the loss function
    def __loss_function(self,Y_tensor, lambda_tv):
        conv_torch_new = pty_torch(Y_tensor, self.psize, self.pscaling,self.sigma)
        B_pred = conv_torch_new.sample_from_convolution(self.skip)
        mse_loss = torch.sum((B_pred - self.B) ** 2)
        # Add TV regularization term
        tv_loss = lambda_tv * self.__total_variation(Y_tensor)
        total_loss = mse_loss + tv_loss
        return total_loss

    
    def run_SGD(self,iterations,learning_rate):

        loss_history = []
        SGD_optimizer = optim.SGD([self.Y_tensor], lr=learning_rate)
        converged = False
        loss_minus = float('inf')
        self.data_list = np.zeros((iterations, 3))
        frob_norm_prev = float('inf')

        for it in tqdm.tqdm(range(iterations)):
            loss = self.__loss_function(self.Y_tensor,self.tv_factor)
            loss_history.append(loss.item())
            SGD_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            SGD_optimizer.step()
            frob_norm = (torch.norm(self.Y_tensor - self.X))
            #Copy the Y_tensor_current to be self.Y_tensor
            
            #Must detach frob_norm such that its numpy value can be accessed
            self.data_list[it,:] = [it,loss.item(),frob_norm.detach().numpy()] 
            if frob_norm <= frob_norm_prev:
                self.Y_tensor_current = self.Y_tensor
                frob_norm_prev = frob_norm
        print("Iteration complete ;)")


        return self.Y_tensor,self.Y_tensor_current
    
    def plot_convergence(self):

        if self.data_list[0,0] is None:
            raise ValueError("Please run the SGD algorithm first")
        
        idx_lowest_norm = np.argmin(self.data_list[:,2])

        fig,axs = plt.subplots(1,2,figsize=(10,4))

        axs[0].plot(self.data_list[:,0],self.data_list[:,1],label='Loss')
        axs[0].set_title("Convergence of Loss")
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(self.data_list[:,0],self.data_list[:,2],label='Frobenius Norm')
        #I want to use the iteration number in this label
        axs[1].plot(self.data_list[idx_lowest_norm,0],self.data_list[idx_lowest_norm,2],'ro',label=f'Lowest Frobenius Norm {idx_lowest_norm}')
        axs[1].set_title("Convergence of Frobenius Norm")
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Frobenius Norm")
        axs[1].legend()
        axs[0].grid()
        axs[1].grid()
        plt.show()

    def plot_final(self):
        if self.data_list[0,0] is None:
            raise ValueError("Please run the SGD algorithm first")
        Y_optimized = self.Y_tensor.detach().numpy()
        Y_optim_smallest_norm = self.Y_tensor_current.detach().numpy()

        # Make big subplot with B, original image, Y_optimized, and Y_optimized_denoised
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))  # Adjust figsize as needed

        im1 = ax[0].imshow(self.B_np, cmap='gray')
        ax[0].set_title('B')
        fig.colorbar(im1, ax=ax[0])

        im2 = ax[1].imshow(Y_optimized, cmap='gray')
        ax[1].set_title('Y_optimized')
        fig.colorbar(im2, ax=ax[1])

        im3 = ax[2].imshow(Y_optim_smallest_norm, cmap='gray')
        ax[2].set_title('Y_optim_smallest_norm')
        fig.colorbar(im3, ax=ax[2])

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def plot_final_two(self):
        if self.data_list[0,0] is None:
            raise ValueError("Please run the SGD algorithm first")

        Y_optim_smallest_norm = self.Y_tensor_current.detach().numpy()

        # Make big subplot with B, original image, Y_optimized, and Y_optimized_denoised
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))  # Adjust figsize as needed

        im1 = ax[0].imshow(self.B_np, cmap='gray')
        ax[0].set_title('B')
        fig.colorbar(im1, ax=ax[0])

        im2 = ax[1].imshow(Y_optim_smallest_norm, cmap='gray')
        ax[1].set_title('Y_optimized')
        fig.colorbar(im2, ax=ax[1])

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    
    



