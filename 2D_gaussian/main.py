import numpy as np
import matplotlib.pyplot as plt
import torch

mean = np.array([0., 0.])
cov = np.array([[ 1. , 0.8], [0.8,  1.]])

target = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.from_numpy(mean), precision_matrix = torch.from_numpy(cov))
result = target.sample(sample_shape = torch.Size([100]))





plt.scatter(result[:,0],result[:,1])


plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.yticks([])

plt.xlabel('S',labelpad=20)
plt.ylabel('A',labelpad=20)
plt.axis()

plt.savefig("pic2.png")