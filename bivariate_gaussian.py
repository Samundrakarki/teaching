%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# generate 2d Gaussian distribution 
mu = [0,0]
Sigma = [[1,0],[0,1]]
rv = multivariate_normal(mu, Sigma)

# plot PDF
x_1 = np.linspace(-5,5,500)
x_2 = np.linspace(-5,5,500)
X_1, X_2 = np.meshgrid(x_1,x_2)
pos = np.empty(X_1.shape + (2,))
pos[:, :, 0] = X_1; pos[:, :, 1] = X_2
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X_1, X_2, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.set_zlabel('density')
plt.show()
