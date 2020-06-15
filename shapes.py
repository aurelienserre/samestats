import torch
import math


"""Examples of shapes for testing"""

center_point = torch.tensor([[.5, .5]])
two_points = torch.tensor([[.25, .5], [.75, .5]])
one_line = torch.tensor([[[.75, .25], [.25, .75]]])
three_lines = torch.tensor([[[.33, 1.], [1, .33]],
                            [[.66, 0.], [0., .66]],
                            [[.33, .33], [.66, .66]]])


def star(n=5):
    angles = torch.tensor([2 * math.pi * i / n for i in range(n)]).unsqueeze(1) + math.pi / 2
    center = torch.tensor([.5, .5])
    ext_points = center + .5 * torch.cat((torch.cos(angles),
                                          torch.sin(angles)), dim=1)
    int_points = center + .25 * torch.cat((torch.cos(angles + math.pi / n),
                                           torch.sin(angles + math.pi / n)), dim=1)
    lines = torch.cat((torch.stack((ext_points, int_points), dim=1),
                       torch.stack((ext_points, int_points.roll(1, 0)), dim=1)))

    return lines
