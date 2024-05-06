This Software is for blurring and deblurring. 


This software falls into two categories. An image can be blurred and deblurred using the 'Script.py' python file. This uses TORCH Stochastic gradient descent to recover the original image. Note the probe is mapped over the image using a large for loop 'Naive' structure. 
Hence this model takes a long time to run for large Probe sizes. In fact it will be O(N^2M^2) (Roughly) in which N and M are the size of the image and probe respectively. The reasons I used this large for loop was because the skip value could be adjusted.

You can define the skip value, image_path, number of iterations for the image recovery.


The 'Conv_matrix' folder contains the software to place the probe using a large Topelitz matrix which will define where the probe will be placed. The 'Script.py' can be run here, within this folder the user can choose to use a gaussian matrix of size m
In which m is odd. Note that the user can also sharpen the image using a 3X3 sharpening topelitz matrix. 


You will need CV2, Matplotlib, PyTorch downloaded. 
