from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

X = [[0], [1], [2], [3], [4]]
y = [0, 0.3, 0.75, 1, 2]

lin_regr = linear_model.LinearRegression()
lin_regr.fit(X, y)

X_eval = np.linspace(0,4,1000)
X_eval = X_eval.reshape(-1,1)

plt.figure()
plt.plot(X_eval,lin_regr.predict(X_eval), label="linear regression predictor")
plt.plot(X,y, 'rs', markersize=12, label="trainin set")
plt.title("Simplistic test of linear regression on 1d input")
plt.show()
