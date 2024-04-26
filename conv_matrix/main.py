import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pplt
import cv2
import scipy 

image = '/Users/charliecameron/CodingHub/Ptychography_project/Software_pty/conv_matrix/moto.png'

#find dimension of image

X = plt.imread(image)
Xg = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY) 

def make_square(image):
    smallest_dim = [np.argmin(image.shape),np.min(image.shape)]

    dim = smallest_dim[0]
    if dim == 0:
        image_new = image[:,:smallest_dim[1]]
    else:
        image_new = image[:smallest_dim[1],:]
    return image_new

def display_image(image):
# Display the image

    plt.imshow(image, cmap='gray')
    plt.show()

image = make_square(Xg)

#print(image.shape)

#image = np.zeros((4,4))

P_list = np.array([0,-1,0,-1,2,-1,0,-1,0])
P = np.reshape(P_list,(3,3))
print(P)
padded_image = np.pad(image, (1,1), 'constant', constant_values=(0,0))

#A must be a sparse matrix of size : image.shape[0]**2 by padded.image[0]**2
#There are three internal submatrices, P_0,P_1 and P_{-1} size image.shape[0] by padded.image[0]
#P_1 main diagonal, P_0 +1 diagonal, P_{-1} +2 diagonal 

P_1 = np.zeros((image.shape[0],padded_image.shape[0]))
P_0 = np.copy(P_1)
P_M1 = np.copy(P_1)

#start with P_1 (top row of P), fill diagonal
for i in range(P_1.shape[0]):
    """
    Filling in the individual submatrices
    """
    P_1[i,i] = P[0,0]
    P_1[i,i+1] = P[0,1]
    P_1[i,i+2] = P[0,2]

    P_0[i,i] = P[1,0]
    P_0[i,i+1] = P[1,1]
    P_0[i,i+2] = P[1,2]

    P_M1[i,i] = P[2,0]
    P_M1[i,i+1] = P[2,1]
    P_M1[i,i+2] = P[2,2]

N = image.shape[0]
N_tild = padded_image.shape[0]

# Fill the lists with the data and indices
aa = np.zeros(9*N**2,dtype = float) #9 * N Value array
ii = np.zeros(N**2,dtype = int) # N (Number of columns) Start of row (corresponds to aa)
jj = np.zeros(9*N**2, dtype = int) #9 * N Column array 

print(N,"size of the image")

ind = 0
shift = 0
for row in range(N**2): #Range over number of rows


    ii[row] = ind
    diag_col = row + shift
    lateral_shift = padded_image.shape[0]
    

    ##THE P_1 matrix
    aa[ind] = P[0,0]
    jj[ind] = diag_col
    ind += 1 

    aa[ind] = P[0,1]
    jj[ind] = diag_col+1
    ind += 1

    aa[ind] = P[0,2]
    jj[ind] = diag_col + 2
    ind += 1

    ##The P_0 probe
    aa[ind] = P[1,0]
    jj[ind] = diag_col + lateral_shift 
    ind += 1 

    aa[ind] = P[1,1]
    jj[ind] = diag_col+lateral_shift+1
    ind += 1

    aa[ind] = P[1,2]
    jj[ind] = diag_col + lateral_shift+2
    ind += 1
    ##The P_-1 probe
    aa[ind] = P[2,0]
    jj[ind] = diag_col + 2*lateral_shift
    ind += 1 

    aa[ind] = P[2,1]
    jj[ind] = diag_col+2*lateral_shift+1
    ind += 1

    aa[ind] = P[2,2]
    jj[ind] = diag_col +2*lateral_shift+2
    ind += 1
    #should contain 0, 9, 18,...

    if (row+1)%N==0 and row != 0: #THE BUG shifts the block matrix 
        shift += N_tild-N 
    
print(len(aa))
print(len(jj))
print(len(ii)) 

jj = np.minimum(jj, N_tild**2 - 1)

ii = np.append(ii,len(aa))
#matrixA = scipy.sparse.csr_matrix((aa, jj, ii), shape=(N**2,N_tild**2))

#dense_matrixA = matrixA.toarray()

# Print the dense matrix
def display_matrix(matrix):
    for row in matrix:
        print(" ".join(str(element).rjust(4) for element in row))

#display_matrix(dense_matrixA)
 
#print the physical matrix A

# fig, ax = plt.subplots(figsize=(3, 3))
# ax.spy(matrixA)


X_flattened = padded_image.flatten('F')
b = np.zeros(N**2,dtype = float)
print(len(b),"len of b)")
for i in range(N**2):
    start = ii[i]
    end = ii[i+1] if i < N**2 - 1 else len(aa)
    indices = jj[start:end]
    valid_indices = indices[indices < len(X_flattened)]
    b[i] = np.sum(aa[start:start+len(valid_indices)] * X_flattened[valid_indices])


# print(len(b))
# print(image.shape)
blur = np.reshape(b,(N,N),'F')
display_image(blur)
plt.show()


