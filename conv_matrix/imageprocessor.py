import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.sparse

class ImageProcessor:
    def __init__(self, image_path,P):
        self.image_path = image_path
        self.X = plt.imread(image_path)
        self.Xg = cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY)
        self.image = self.make_square(self.Xg)
        self.P = P 
        if P.shape[0] != P.shape[1]:
            raise ValueError("Probe must be square")
        
        self.padded_image = np.pad(self.image, (self.P.shape[0]//2, self.P.shape[0]//2), 'constant', constant_values=(0, 0))

        self.N = self.image.shape[0]
        self.N_tild = self.padded_image.shape[0]
        self.M = self.P.shape[0]
        

    def make_square(self, image):
        """"
        Generate a square image from the original image
        """
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





    def _create_convolution_2(self):
        """
        This is the main loop. THis will output the Topelitz. This depends on the image size and the probe size

        OUTPUT: Filled in Toeplitz matrix A
        """
        p_flat = self.P.flatten()
        p_row_len = self.P.shape[1]  # Assuming p is a 3x3 matrix
        self.aa = np.zeros(self.M**2 * self.N ** 2, dtype=float)
        self.ii = np.zeros(self.N ** 2, dtype=int)
        self.jj = np.zeros(self.M**2 * self.N ** 2, dtype=int)

        ind = 0
        shift = 0

        for row in range(self.N ** 2):
            lateral_shift_counter = 0  # Reset lateral shift counter for each row
            if (row + 1) % self.N == 0 and row != 0:
                shift += self.N_tild - self.N

            for i, element in enumerate(p_flat):
                self.ii[row] = ind
                diag_col = row + shift
                if i % p_row_len == 0 and i != 0:
                    lateral_shift_counter += 1
                self.aa[ind] = element
                self.jj[ind] = diag_col + i%p_row_len + lateral_shift_counter*self.N_tild
                ind += 1

        self.jj = np.minimum(self.jj, self.N_tild ** 2 - 1)
        self.ii = np.append(self.ii, len(self.aa))




    def _create_convolution(self):

        """"
        Creates the convolution matrix
        """
        p_flat = self.P.flatten()
        self.aa = np.zeros(self.M**2 * self.N ** 2, dtype=float) #Number of elements M**2 elements per row
        self.ii = np.zeros(self.N ** 2, dtype=int)
        self.jj = np.zeros(self.M**2 * self.N ** 2, dtype=int)

        ind = 0
        shift = 0
        for row in range(self.N ** 2):
            self.ii[row] = ind
            diag_col = row + shift
            lateral_shift = self.N_tild
            self.aa[ind] = p_flat[0]
            self.jj[ind] = diag_col
            ind += 1

            self.aa[ind] = p_flat[1]
            self.jj[ind] = diag_col + 1
            ind += 1

            self.aa[ind] = p_flat[2]
            self.jj[ind] = diag_col + 2
            ind += 1

            self.aa[ind] = p_flat[3]
            self.jj[ind] = diag_col + lateral_shift
            ind += 1

            self.aa[ind] = p_flat[4]
            self.jj[ind] = diag_col + lateral_shift + 1
            ind += 1

            self.aa[ind] = p_flat[5]
            self.jj[ind] = diag_col + lateral_shift + 2
            ind += 1

            self.aa[ind] = p_flat[6]
            self.jj[ind] = diag_col + 2 * lateral_shift
            ind += 1

            self.aa[ind] = p_flat[7]
            self.jj[ind] = diag_col + 2 * lateral_shift + 1
            ind += 1

            self.aa[ind] = p_flat[8]
            self.jj[ind] = diag_col + 2 * lateral_shift + 2
            ind += 1

            if (row + 1) % self.N == 0 and row != 0:
                shift += self.N_tild - self.N

        self.jj = np.minimum(self.jj, self.N_tild ** 2 - 1)

        self.ii = np.append(self.ii, len(self.aa))

    def create_blur(self):
        
        """Create convolution matrix and multiply with flattened image
         OUTPUT: blur
           """
        self._create_convolution_2()
        X_flattened = self.padded_image.flatten('F')
        b = np.zeros(self.N ** 2, dtype=float)

        for i in range(self.N ** 2):
            start = self.ii[i]
            end = self.ii[i + 1] if i < self.N ** 2 - 1 else len(self.aa)
            indices = self.jj[start:end]
            valid_indices = indices[indices < len(X_flattened)]
            b[i] = np.sum(self.aa[start:start + len(valid_indices)] * X_flattened[valid_indices])

        blur = np.reshape(b, (self.N, self.N), 'F')
        return blur