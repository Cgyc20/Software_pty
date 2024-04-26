import numpy as np
import matplotlib.pyplot as plt

from imageprocessor import ImageProcessor
image_path = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/conv_matrix/moto.png'
image_path_2 = '/Users/charliecameron/CodingHub/Ptychography_project/Sophie_lawn.png'



def gaussian(width, sigma=1, amplitude=1):
    """
    Generate an adjustable Gaussian probe
    """
    x = np.linspace(-width//2, width//2, width)
    y = np.linspace(-width//2, width//2, width)
    X, Y = np.meshgrid(x, y)
    Z = amplitude * np.exp(-0.5*((X**2 + Y**2) / sigma**2))
    return Z
Z = gaussian(7,sigma=10)


S_l = np.array([0,-1,0,-1,4,-1,0,-1,0])
S = S_l.reshape((3,3))
print(S)
Model = ImageProcessor(image_path,S)

orig_image = Model.image

blur_image = Model.create_blur()
#Model.display_image(blur_image)

#Plot original and converted with colour bar

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(orig_image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(blur_image, cmap='gray')
ax[1].set_title('Sharpened')
ax[1].axis('off')

plt.show()


