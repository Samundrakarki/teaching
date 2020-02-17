from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

X = [[0], [1], [2], [3], [4]]
y = [0, 0.3, 0.75, 1, 2]

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y)

X_eval = np.linspace(0,4,1000)
X_eval = X_eval.reshape(-1,1)

plt.figure()
plt.plot(X_eval,neigh.predict(X_eval), label="kNN regression predictor")
plt.plot(X,y, 'rs', markersize=12, label="trainin set")
plt.title("Simplistic test of kNN regression")
plt.show()
