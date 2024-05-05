import numpy as np
import matplotlib.pyplot as plt
import torch 
import cv2

class ImageProcessorTorch:
    def __init__(self,P):
        """Input a probe which is square"""
        self.P = torch.tensor(P, dtype=torch.float32)
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Probe must be square")
        self.M = self.P.shape[0]
        
    def input_image(self,image_path):
        """Let the user input an image path"""

        self.image_path = image_path
        self.X = plt.imread(image_path)
        self.Xg = torch.tensor(cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY), dtype=torch.float32)
        self.image = self.make_square(self.Xg)
        self.padded_image = torch.nn.functional.pad(self.image, (self.P.shape[0]//2, self.P.shape[0]//2))

        self.N = self.image.shape[0]
        self.N_tild = self.padded_image.shape[0]

    def input_matrix(self,X):
        self.image = self.make_square(torch.tensor(X, dtype=torch.float32))
        self.padded_image = torch.nn.functional.pad(self.image, (self.P.shape[0]//2, self.P.shape[0]//2))
        self.N = self.image.shape[0]
        self.N_tild = self.padded_image.shape[0]

    def make_square(self, image):
        """
        Generate a square image from the original image
        """
        if len(image.shape) != 2:
            raise ValueError ("Image must be two dimensional")
        smallest_dim = min(image.shape)
        image_new = image[:smallest_dim, :smallest_dim]
        return image_new

    def display_image(self, image):
        plt.imshow(image.numpy(), cmap='gray')
        plt.show()


    def create_convolution_2(self):
        """
        This is the main loop. THis will output the Topelitz. This depends on the image size and the probe size

        OUTPUT: Filled in Toeplitz matrix A
        """
        p_flat = self.P.flatten()
        p_row_len = self.P.shape[0]  # Assuming p is a 3x3 matrix
        self.aa = torch.zeros(self.M**2 * self.N ** 2, dtype=torch.float32)
        self.ii = torch.zeros(self.N ** 2, dtype=torch.int64)
        self.jj = torch.zeros(self.M**2 * self.N ** 2, dtype=torch.int64)

        ind = 0
        shift = 0

        for row in range(self.N ** 2):
            lateral_shift_counter = 0  # Reset lateral shift counter for each row
            self.ii[row] = ind

            if (row) % self.N == 0 and row != 0: 
                shift += self.N_tild - self.N

            for i, element in enumerate(p_flat):
                diag_col = row + shift
                if i % p_row_len == 0 and i != 0:
                    lateral_shift_counter += 1
                self.aa[ind] = element
                self.jj[ind] = diag_col + i % p_row_len + (lateral_shift_counter) * self.N_tild
                ind += 1

        self.jj = torch.minimum(self.jj, torch.tensor(self.N_tild ** 2 - 1, dtype=torch.int64))
        self.ii = torch.cat((self.ii, torch.tensor([len(self.aa)], dtype=torch.int64)))

    def create_blur(self):
        """Create convolution matrix and multiply with flattened image
         OUTPUT: blur 
        """
        self.create_convolution_2()
        X_flattened = self.padded_image.flatten()
        b = torch.zeros(self.N ** 2, dtype=torch.float32)

        for i in range(self.N ** 2):
            start = self.ii[i]
            end = self.ii[i + 1] if i < self.N ** 2 - 1 else len(self.aa)
            indices = self.jj[start:end]
            valid_indices = indices[indices < len(X_flattened)]
            b[i] = torch.sum(self.aa[start:start + len(valid_indices)] * X_flattened[valid_indices])

        blur = torch.reshape(b, (self.N, self.N))
        return blur
