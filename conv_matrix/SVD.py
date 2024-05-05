import numpy as np
import matplotlib.pyplot as plt
import scipy


from ImageProcessorPeriod import ImageProcessorPeriod
image_path = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/conv_matrix/moto.png'
image_path_2 = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/conv_matrix/moto_comp.JPEG'



def gaussian(width, sigma=1):
    """
    Generate an adjustable Gaussian probe
    """
    x = np.linspace(-width//2, width//2, width)
    y = np.linspace(-width//2, width//2, width)
    X, Y = np.meshgrid(x, y)
    Z =  np.exp(-0.5*((X**2 + Y**2) / sigma**2))
    return Z
Z = gaussian(3,sigma=10)

matrix = np.ones((3,3))
Model = ImageProcessorPeriod(Z)

Model.input_image(image_path=image_path_2)
#Model.input_matrix(matrix)
orig_image = Model.image
blur_image = Model.create_blur()
#Model.plot_sparse()

Model.display_image(blur_image)
#np.linalg.svd
aa = Model.aa
ii = Model.ii
jj = Model.jj


print(len(aa))
print(len(ii))
print(len(jj))


A = scipy.sparse.csr_matrix((aa, jj, ii), shape=(Model.N**2, Model.N**2))

# Perform sparse SVD
U, s, V = scipy.sparse.linalg.svds(A, k=min(A.shape)-1)

# Reconstruct the dense matrices
S = np.diag(s)
#get inverse of S
S_inv = np.linalg.inv(S)

intermediate = np.dot(np.dot(U, S_inv), V)

print(np.shape(intermediate))
print(np.shape(blur_image.flatten('F')))


output = np.dot(intermediate, blur_image.flatten('F'))
output = output.reshape((Model.N,Model.N))

Model.display_image(output)








