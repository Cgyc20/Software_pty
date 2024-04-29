import numpy as np
import matplotlib.pyplot as plt
from imageprocessor import ImageProcessor


image = np.ones((3,3))
p = np.array([1,-1,1,-1,4,-1,1,-1,1])
probe = p.reshape((3,3))
#probe = np.ones((3,3))



Model = ImageProcessor(probe)
Model.input_matrix(image)
#Model.input_image('/Users/charliecameron/CodingHub/Ptychography_project/Sophie_lawn.png')
BLUR = Model.create_blur()
#Model.display_image(BLUR)
print(Model.padded_image)
Model.plot_sparse()



