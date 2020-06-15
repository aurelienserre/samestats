import torch
import torch.nn as nn


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

        points = torch.unsqueeze(points, dim=1)     # (n_points, 1, n_coords)
        sq_diff = (self.targets - points) ** 2      # (n_points, n_targets, n_coords)
        dist_mat = torch.sqrt((sq_diff).sum(dim=2))     # (n_points, n_targets)
        dist_to_closest = torch.min(dist_mat, dim=1).values  # (n_points,)
        sum_of_dists = dist_to_closest.sum()     # scalar

        return sum_of_dists


class DistToLines2D(nn.Module):

    """Compute the sum over all the points given as input of
    their squared distances to some target lines. Only for 2D datasets"""

    def __init__(self, lines):
        """Initializes the target lines

        :lines: (np.array or torch.Tensor) target lines defines by two points
            dimensions: (n_targets, 2, n_coords=2)

        """
        super(DistToLines2D, self).__init__()

        self.lines = torch.Tensor(lines)

    def forward(self, points):
        """Computes the sum over all points of the squared distance to the
        closest line"""

        # let M a point from the dataset, and (P1, P2) the two points defining a target line.
        # P1P2 = P2 - P1
        P1P2 = self.lines[:, 1, :] - self.lines[:, 0, :]    # (n_targets, 2)
        # norm of P1P2
        seg_norm = torch.sqrt((P1P2 ** 2).sum(dim=1))     # (n_targets,)
        # P1M = M - P1, P2M = M - P2
        P1M = points[:, None, :] - self.lines[:, 0, :]      # (n_points, n_targets, 2)
        P2M = points[:, None, :] - self.lines[:, 1, :]      # (n_points, n_targets, 2)
        # dot product P1M . P1P2
        dot_prod = torch.matmul(P1P2[:, None, :],
                                P1M[:, :, :, None]).squeeze()   # (n_points, n_targets)
        # shortest distance from M to P1 or P2
        dist_closest = torch.min(torch.sqrt((P1M ** 2).sum(dim=-1)),
                                 torch.sqrt((P2M ** 2).sum(dim=-1)))    # (n_points, n_targets)
        # projection of M on (P1P2)
        H = self.lines[:, 0, :] \
            + (dot_prod / (seg_norm ** 2)).unsqueeze(dim=-1) * P1P2   # (n_points, n_targets, 2)
        # distance from M to its projection H
        MH = H - points.unsqueeze(dim=1)    # (n_points, n_targets, 2)
        dist_proj = torch.sqrt((MH ** 2).sum(dim=-1))    # (n_points, n_targets)
        # dist from M to segment P1P2 = dist_proj if H falls on the segment
        # P1P2, or dist_closest otherwise
        dist = torch.where((0 < dot_prod) & (dot_prod < (seg_norm) ** 2),
                           dist_proj, dist_closest)   # (n_points, n_targets)
        dist_to_closest = torch.min(dist, dim=1).values     # (n_points,)
        sum_of_dist = dist_to_closest.sum()     # scalar

        return sum_of_dist
