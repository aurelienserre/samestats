import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.optim as optim
from modules import Dataset, DistToPoints


# torch.set_printoptions(precision=5)
# torch.random.manual_seed(5367838)

n_samples = 100
points = torch.empty(n_samples, 2).uniform_(0, 1)
ds = Dataset(points)
targets = torch.tensor([[.25, .5], [.75, .5]])
sod = DistToPoints(targets)

optimizer = optim.SGD(ds.parameters(), lr=1e-1)


def iteration(i):
    optimizer.zero_grad()

    error = sod(ds())
    error.backward()
    optimizer.step()

    particules = ds.points.detach().numpy()

    plt.cla()
    plt.scatter(particules[:, 0], particules[:, 1])
    plt.xlim((0, 1))
    plt.ylim((0, 1))


ani = FuncAnimation(plt.gcf(), iteration, interval=100)
plt.show()
