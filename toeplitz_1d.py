import numpy as np
import matplotlib.pyplot as plt

length = 100
step_size = 20
x = np.zeros(length)
mid_point = length // 2
start_point = mid_point - (step_size // 2)
end_point = start_point + step_size
x[start_point:end_point] = 1

print(x)

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

probe_length = 25
mu = probe_length // 2  # Center of the Gaussian curve
sigma = 1  # Standard deviation of the Gaussian curve

# Create an array of indices representing the positions in the probe
indices = np.arange(probe_length)

# Generate the Gaussian curve
probe = gaussian(indices, mu, sigma)
probe = probe/np.sum(probe)
# Pad the input signal
x_padded = np.zeros(len(x)+len(probe)-1)
x_padded[len(probe)//2:len(x)+len(probe)//2] = x

# Initialize the probe matrix
probe_matrix = np.zeros((len(x), len(x_padded)))
# Construct the Toeplitz matrix with probe entries


for i in range(probe_matrix.shape[0]):
    probe_matrix[i,i:len(probe)+i] = probe


b = np.matmul(probe_matrix, x_padded)

plt.figure()
plt.plot(x, label='x')
plt.plot(b, 'r--',label='b')
plt.legend(fontsize='x-large')  # Increase the fontsize of the legend
plt.xlabel('Index', fontsize='x-large')  # Increase the fontsize of the x-label
plt.ylabel('Intensity', fontsize='x-large')  # Increase the fontsize of the y-label

# Increase the fontsize of the tick labels
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')

plt.grid()
plt.show()

#plot the probe vector

plt.figure()
plt.plot(probe)
plt.xlabel('Index', fontsize='x-large')  # Increase the fontsize of the x-label
plt.ylabel('Intensity', fontsize='x-large')  # Increase the fontsize of the y-label
plt.show()