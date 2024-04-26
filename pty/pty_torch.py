import torch

class pty_torch:

    def __init__(self, X, P_size, scaling,sigma):
        self.X = X
        self.scaling = scaling
        self.sigma = sigma

        if P_size % 2 == 0:
            raise Exception("Length of Probe must be odd")
        self.P_size = P_size
        self.Probe = self._radius()

    """ def _radius(self):
            Probe = torch.zeros((self.P_size, self.P_size))
            radius_value = (self.P_size + 1) // 2

            for i in range(Probe.shape[0]):
                for j in range(Probe.shape[1]):
                    i_torch = torch.tensor(i, dtype=torch.float32)
                    j_torch = torch.tensor(j, dtype=torch.float32)
                    distance = torch.sqrt((i_torch - radius_value + 1) ** 2 + (j_torch - radius_value + 1) ** 2)
                    if distance <= radius_value * self.scaling:
                        sigma = radius_value * self.scaling * 1.0
                        Probe[i, j] = torch.exp((-distance ** 2) / (2 * sigma ** 2))

            return Probe"""
    

    def _radius(self):
        Probe = torch.zeros((self.P_size, self.P_size))
        radius_value = (self.P_size + 1) // 2
        #sigma = radius_value * self.scaling * 1.0

        # Create a grid of indices
        indices = torch.arange(start=0, end=self.P_size, dtype=torch.float32)
        i_torch, j_torch = torch.meshgrid(indices, indices)
        # Calculate the distance from the center for each point in the grid
        distance = torch.sqrt((i_torch - radius_value + 1) ** 2 + (j_torch - radius_value + 1) ** 2)

        # Apply the condition and calculate the Probe values in a vectorized way
        mask = distance <= radius_value * self.scaling
        Probe[mask] = torch.exp((-distance[mask] ** 2) / (2 * self.sigma ** 2))

        return Probe

    def _convolution_padding(self):
        X_padded = torch.nn.functional.pad(self.X, (self.P_size // 2, self.P_size // 2, self.P_size // 2, self.P_size // 2), mode='constant', value=0)
        P_flip = torch.flip(torch.flip(self.Probe, [0]), [1])
        return P_flip, X_padded

    def convolution_process(self):
        P_flip, X_padded = self._convolution_padding()
        output_image = torch.nn.functional.conv2d(X_padded.unsqueeze(0).unsqueeze(0), P_flip.unsqueeze(0).unsqueeze(0), padding=0).squeeze()
        return output_image

    def sample_from_convolution(self, skip):
        full_output = self.convolution_process()
        row_indices = torch.arange(0, full_output.size(0), skip)
        col_indices = torch.arange(0, full_output.size(1), skip)
        subsampled_matrix = full_output[row_indices[:, None], col_indices]
        return subsampled_matrix
