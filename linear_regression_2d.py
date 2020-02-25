from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.array([[0, 0], [1,2], [2,4], [3,0], [4,1]])
y = np.array([0, 0.3, 0.75, 1, 2])

lin_regr = linear_model.LinearRegression()
lin_regr.fit(X, y)

fig = plt.figure()#(figsize=(3, 2))
ax = Axes3D(fig, elev=45, azim=-120)
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label='training set')
ax.plot_surface(np.array([[-1, -1], [5, 5]]), np.array([[-1, 5], [-1, 5]]), lin_regr.predict(np.array([[-1, -1, 5, 5], [-1, 5, -1, 5]]).T).reshape((2, 2)), alpha=.5, label='linear regression predictor')
ax.set_xlabel('input X_1')
ax.set_ylabel('input X_2')
ax.set_zlabel('output Y')
plt.title("Simplistic test of linear regression on 2d input")
plt.show()
