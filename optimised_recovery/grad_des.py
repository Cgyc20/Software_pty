import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import tqdm

class grad_des_SGD:
     
    def __init__(self,Model,B,X, tv_factor, randomised=False):
        torch.set_grad_enabled(True) 
        
        self.B = B
        print(type(B))
        self.X = X
        self.tv_factor = tv_factor
        self.Model = Model
        # Get the sampled matrix B
        self.B_np = B
        self.Y_tensor = None
        if randomised:
            self.Y_tensor = torch.randn(self.X.shape, dtype=torch.float32, requires_grad=True)
            print("Using entirely random Initial matrix")
        else:
            pass

    def random_create(self, factor):
        self.Y_tensor = torch.tensor(self.X + factor * torch.randn(self.X.shape, dtype=torch.float32), requires_grad=True)

    def _total_variation(self, image):
        # Compute finite differences along height and width dimensions
        dh = image[1:, :] - image[:-1, :]
        dw = image[:, 1:] - image[:, :-1]
        # Compute total variation
        tv = torch.sum(torch.abs(dh)) + torch.sum(torch.abs(dw))
        return tv

    def _loss_function(self, Y_tensor, lambda_tv):
        
        B_pred = self.Model.create_blur()
        mse_loss = torch.sum((B_pred - self.B) ** 2)
        # Add TV regularization term
        tv_loss = lambda_tv * self._total_variation(Y_tensor)
        total_loss = mse_loss + tv_loss
        return total_loss

    def run_SGD(self, iterations, learning_rate):

        loss_history = []
        SGD_optimizer = optim.SGD([self.Y_tensor], lr=learning_rate)

        for it in tqdm.tqdm(range(iterations)):
            self.loss = self._loss_function(self.Y_tensor, self.tv_factor)
            self.loss_history.append(self.loss.item())
            SGD_optimizer.zero_grad()
            self.loss.backward(retain_graph=True)
            SGD_optimizer.step()

        print("Iteration complete ;)")
        return self.Y_tensor

    def plot_convergence(self):

        if self.Y_tensor is None:
            raise ValueError("Please run the SGD algorithm first")

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].plot(range(len(self.loss_history)), self.loss_history, label='Loss')
        axs[0].set_title("Convergence of Loss")
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(range(len(self.loss_history)), self.loss_history, label='Frobenius Norm')
        # I want to use the iteration number in this label
        axs[1].set_title("Convergence of Frobenius Norm")
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Frobenius Norm")
        axs[1].legend()
        axs[0].grid()
        axs[1].grid()
        plt.show()

    def plot_final(self):
        if self.Y_tensor is None:
            raise ValueError("Please run the SGD algorithm first")
        Y_optimized = self.Y_tensor.detach().numpy()

        # Make big subplot with B, original image, Y_optimized
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))  # Adjust figsize as needed

        im1 = ax[0].imshow(self.B_np, cmap='gray')
        ax[0].set_title('B')
        fig.colorbar(im1, ax=ax[0])

        im2 = ax[1].imshow(Y_optimized, cmap='gray')
        ax[1].set_title('Y_optimized')
        fig.colorbar(im2, ax=ax[1])

        im3 = ax[2].imshow(self.X, cmap='gray')
        ax[2].set_title('Original Image')
        fig.colorbar(im3, ax=ax[2])

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
