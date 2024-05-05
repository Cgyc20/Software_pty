import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.sparse

class ImageProcessorPeriod:
    def __init__(self,P):
        """Input a probe which is square"""
        self.P = P 
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Probe must be square")
        self.M = self.P.shape[0]
        
    def input_image(self,image_path):
        """Let the user input an image path"""

        self.image_path = image_path
        self.X = plt.imread(image_path)
        self.Xg = cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY)
        self.image = self.make_square(self.Xg)

        self.N = self.image.shape[0]

    def input_matrix(self,X):
        self.image = self.make_square(X)
        self.N = self.image.shape[0]

    def make_square(self, image):
        """
        Generate a square image from the original image
        """
        if len(np.shape(image)) !=2:
            raise ValueError ("Image must be two dimensiona")
        smallest_dim = [np.argmin(image.shape), np.min(image.shape)]
        dim = smallest_dim[0]
        if dim == 0:
            image_new = image[:, :smallest_dim[1]]
        else:
            image_new = image[:smallest_dim[1], :]
        return image_new

    def display_image(self, image):
        plt.imshow(image, cmap='gray')
        plt.show()


    def create_convolution_2(self):
        """
        This is the main loop. THis will output the Topelitz. This depends on the image size and the probe size

        OUTPUT: Filled in Toeplitz matrix A
        """
        p_flat = self.P.flatten()
        p_row_len = self.P.shape[0]  # Assuming p is a 3x3 matrix
        self.aa = np.zeros(self.M**2 * self.N ** 2, dtype=float)
        self.ii = np.zeros(self.N ** 2, dtype=int)
        self.jj = np.zeros(self.M**2 * self.N ** 2, dtype=int)

        ind = 0
        shift = 0

        for row in range(self.N ** 2):
            self.ii[row] = ind

            # #if (row + 1 ) % self.N == 0 and row != 0: THIS WAS BUG ALSO
            # if (row ) % self.N == 0 and row != 0: 
            #     shift += self.N
            

            for i, element in enumerate(p_flat):
                #self.ii[row] = ind THIS WAS THE BUG 
                diag_col = row + shift
                self.aa[ind] = element
                self.jj[ind] = diag_col + i%p_row_len 
                ind += 1

        self.jj = np.minimum(self.jj, self.N ** 2 - 1)
        self.ii = np.append(self.ii, len(self.aa))


    def plot_sparse(self):
        """Plot the sparse matrix"""
        self.create_convolution_2()
        #self.jj = np.minimum(self.jj, self.N_tild**2 - 1)  # Ensure all column indices are within bounds
        self.ii[0] = 0  # Ensure the first element of ii is zero
        matrixA = scipy.sparse.csr_matrix((self.aa, self.jj, self.ii), shape=(self.N**2,self.N**2))
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figure size here
        ax.spy(matrixA)
        plt.show()



    def create_blur(self):
        
        """Create convolution matrix and multiply with flattened image
         OUTPUT: blur 
           """
        self.create_convolution_2()
        X_flattened = self.image.flatten('F')
        b = np.zeros(self.N ** 2, dtype=float)

        for i in range(self.N ** 2):
            start = self.ii[i]
            end = self.ii[i + 1] if i < self.N ** 2 - 1 else len(self.aa)
            indices = self.jj[start:end]
            valid_indices = indices[indices < len(X_flattened)]
            b[i] = np.sum(self.aa[start:start + len(valid_indices)] * X_flattened[valid_indices])

        blur = np.reshape(b, (self.N, self.N), 'F')
        return blur