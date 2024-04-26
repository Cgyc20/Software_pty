import numpy as np
import matplotlib.pyplot as plt 
import math

class pty_functions:

    def __init__(self, X, P_size,scaling,method,sigma, grayscale=False,strategy = 'extrapolate'):
           
        self.X = X
        self.P_size = P_size
        self.X_gray = self.X if not grayscale else self._grayscale()
        self.scaling = scaling
        self.method = method
        self.sigma = sigma
        if self.P_size % 2 == 0:
            raise Exception("Length of Probe must be odd")
        self.Probe = self._radius()
        self.strategy = strategy
        print(f"Convolution of the image with Probe size {self.P_size}, Scaling factor {self.scaling} with probe values taking {self.method} values")


    def _radius(self): #This generates a circular Beam . Which is then applied onto the pixels
        """
        Generate circular beam based on the probe size and method.

        Returns:
            numpy.ndarray: Circular beam.
        """
        
        Probe = np.zeros((self.P_size,self.P_size))
        radius_value = (self.P_size+1)// 2

        if self.method.lower() == 'constant':
            
            for i in range(Probe.shape[0]):
                for j in range(Probe.shape[1]):
                    distance = np.sqrt((i-radius_value+1)**2+(j-radius_value+1)**2)
                    if distance <= radius_value*self.scaling:
                        Probe[i,j] = 1

        elif self.method.lower() == 'gaussian':
            print(f"Creating Gaussian circular beam with scaling factor {self.scaling}")

            # for i in range(Probe.shape[0]):
            #     for j in range(Probe.shape[1]):
            #         distance = np.sqrt((i-radius_value+1)**2+(j-radius_value+1)**2)
                    # if distance <= radius_value*self.scaling:
                    #     sigma = radius_value*self.scaling*0.4 #This gives a nice gaussian probe 
                    #     Probe[i,j] = np.exp((-distance**2)/(2*sigma**2))


            indices = np.arange(0, self.P_size, dtype=np.float32)
            i_ind, j_ind = np.meshgrid(indices, indices)
            # Calculate the distance from the center for each point in the grid
            distance = np.sqrt((i_ind - radius_value + 1) ** 2 + (j_ind - radius_value + 1) ** 2)

            # Apply the condition and calculate the Probe values in a vectorized way
            mask = distance <= radius_value * self.scaling
            reciprocal = 1/(self.sigma*np.sqrt(2*math.pi))
            Probe[mask] = reciprocal*np.exp((-distance[mask] ** 2) / (2 * self.sigma ** 2))


                        
        else:
            raise Exception("Check the input- Must be either \'gaussian\' or \'constant\'")
        return Probe
    
    

    def _convolution_padding(self):
        strategy = self.strategy.lower()
        X_rows, X_cols = self.X_gray.shape

        if self.P_size > X_rows or self.P_size > X_rows: #The only one to check is that the Probe isn't larger than the image
            print('The probe can\'t be larger than the image')
            return None

        P_pad = self.P_size // 2  # The floor division of the rows, this is the additional padding on the image
        B = np.zeros_like(self.X_gray)  # set up the B matrix which will contain the convolution sum.

        if strategy == 'zeros':
            X_padded = np.pad(self.X_gray, ((P_pad, P_pad), (P_pad, P_pad)), mode='constant', constant_values=0)
        elif strategy == 'extrapolate':
            X_padded = np.pad(self.X_gray, ((P_pad, P_pad), (P_pad, P_pad)), mode='edge')
        else:
            print('Invalid strategy: Please use either \'zeros\' or \'extrapolate\'')
            return None
        P_flip = np.flip(np.flip(self.Probe, axis=1), axis=0)
        return P_flip, X_padded, P_pad


    def convolution_process(self):
        
        """Convolution process"""
        P_flip, X_padded, _ = self._convolution_padding()
        X_rows, X_cols = self.X_gray.shape
        output_image = np.zeros_like(self.X_gray)
        fourier_output = np.zeros_like(self.X_gray)
        for i in range(X_rows):
            for j in range(X_cols):
                X_sample = X_padded[i:i+self.P_size, j:j+self.P_size]
                dot_product = np.dot(P_flip, X_sample)

                output_image[i, j] = np.sum(dot_product)/(self.P_size**2)
                #Now apply fourier transform to each element in the dot_product. And sum
                fourier_output[i,j] = np.abs(np.sum(np.fft.fft2(dot_product)))**2
        self.X_padded = X_padded
        self.fourier_output = fourier_output
        
        return output_image
    
    def sample_from_convolution(self,skip):
        full_output = self.convolution_process()  # Get the full output from convolution
        n, m = full_output.shape
        skip_value = skip  # Calculate the skip value
        # Generate row and column indices for subsampling
        row_indices = np.arange(0, n, skip_value)
        col_indices = np.arange(0, m, skip_value)
        # Subsample the matrix
        subsampled_matrix = full_output[row_indices[:,None], col_indices]
        self.subsampled_fourier = self.fourier_output[row_indices[:,None], col_indices]
        self.subsampled_matrix = subsampled_matrix

        dotted_place = np.zeros_like(full_output)
        dotted_place[row_indices[:,None], col_indices] = 1

        return subsampled_matrix#, dotted_place



    def plot(self,image_to_plot):
        plt.figure(figsize=(15, 5))  # Adjust the figure size as needed
        # Plot the first subplot (Original image)
        plt.subplot(1, 3, 1)
        plt.imshow(self.Probe, cmap='gray')
        # Plot the second subplot (After sampling)
        plt.subplot(1, 3, 2)
        plt.imshow(self.X_gray, cmap='gray')
        # Plot the third subplot (Your third subplot here)
        plt.subplot(1, 3, 3)
        plt.imshow(image_to_plot, cmap='gray')
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

    
    ''' These were some of my ideas that didn't come into fruition 

    def convolution_process_skip(self):
        P_flip, X_padded, _ = self._convolution_padding()
        X_rows, X_cols = self.X_gray.shape

        output_image = np.zeros((X_rows//2+1,X_cols//2+1))

        for i in range(0,X_rows,2):
            for j in range(0,X_cols,2):
                X_sample = X_padded[i:i+self.P_size, j:j+self.P_size]
                output_image[i//2, j//2] = np.sum(np.dot(P_flip, X_sample))/(self.P_size**2)
        self.X_padded = X_padded
        

        return output_image
    
    def convolution_process_skip_four(self):
        P_flip, X_padded, _ = self._convolution_padding()
        X_rows, X_cols = self.X_gray.shape
        output_image = np.zeros((X_rows//4+1,X_cols//4+1))

        for i in range(0,X_rows,4):
            for j in range(0,X_cols,4):
                X_sample = X_padded[i:i+self.P_size, j:j+self.P_size]
                output_image[i//4, j//4] = np.sum(np.dot(P_flip, X_sample))/(self.P_size**2)
        self.X_padded = X_padded
        

        return output_image

    def convolution_skipped(self,pixel_skip):
        P_flip = np.flip(np.flip(self.Probe, axis=1), axis=0) #Flip the probe as before
        X_rows, X_cols = self.X_gray.shape
        P_flip, X_padded, P_pad = self._convolution_padding()
        reference_image = np.zeros_like(X_padded) #The padded matrix

        row_sample_number = X_rows//pixel_skip
        column_sample_number = X_cols//pixel_skip

        row_positions, column_positions = np.meshgrid(np.arange(1, X_rows-1, pixel_skip),
        np.arange(1, X_cols-1, pixel_skip),
        indexing='ij') 

        for i in range(len(row_positions)):
            for j in range(len(column_positions)):
                reference_image[row_positions[i]+P_pad, column_positions[j]+P_pad] = 1

        column_positions, row_positions = np.meshgrid(
        np.arange(0, X_rows, row_sample_number) + self.Probe[0],
        np.arange(0, X_cols, column_sample_number)+self.Probe[0],
        indexing='ij') 

        reference_image = np.zeros_like(X_padded)

        for i in range(row_positions.shape[0]):
            for j in range(column_positions.shape[1]):
                row_idx = row_positions[i, j] - self.Probe[0]//2
                col_idx = column_positions[i, j] - self.Probe[0]//2
                reference_image[row_idx:row_idx+self.Probe[0], col_idx:col_idx + self.Probe[0]] = self.Probe


        return reference_image
      
        '''





