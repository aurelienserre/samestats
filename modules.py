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


class DistToPoints(nn.Module):

    """Compute the sum over all the points given as input of
    their squared distances to some target points."""

    def __init__(self, targets):
        """Initializes the targets

        :targets: (np.array or torch.Tensor) target points
            dimensions: (n_targets, n_coords_of_points)

        """
        super(DistToPoints, self).__init__()

        self.targets = torch.Tensor(targets)

    def forward(self, points):
        """Computes the sum over all points of the squared distance to the
        closest target"""

        points = torch.unsqueeze(points, dim=1)     # (s_points, 1, n_coords)
        sq_diff = (self.targets - points) ** 2      # (n_points, n_targets, n_coords)
        sq_dists = (sq_diff).sum(dim=2)             # (n_points, n_targets)
        sq_dist_to_closest = torch.min(sq_dists, dim=1).values    # (n_points,)
        sum_of_dists = sq_dist_to_closest.sum()     # scalar

        return sum_of_dists


class DistToLines(nn.Module):

    """Compute the sum over all the points given as input of
    their squared distances to some target lines."""

    def __init__(self, lines):
        """Initializes the target lines

        :lines: (np.array or torch.Tensor) target lines defines by two points
            dimensions: (n_targets, 2, n_coords_of_points)

        """
        super(DistToLines, self).__init__()

        self.lines = torch.Tensor(lines)

    def forward(sekf, points):
        """Computes the sum over all oints of the squared distance to the
        closest line"""

        pass
