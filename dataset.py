import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Dataset(nn.Module):

    """Module for representing the points of the data set."""

    def __init__(self, points):
        """Initializes the Dataset module with the given points

        :points: (np.array or torch.Tensor) points of the dataset
            dimensions: (n_samples, n_coords_of_points)

        """
        super(Dataset, self).__init__()

        self.points = Parameter(torch.Tensor(points))

    def forward(self):
        return self.points
