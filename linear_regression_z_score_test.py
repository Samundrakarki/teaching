#%matplotlib notebook

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(1,2,3,4,5,6,7,8))
Y=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(9))

# fit linear regression by least squares
N=X.shape[0]
D=X.shape[1]
O=np.ones((N,1))
X=np.concatenate((O,X),axis=1)
M = X.T@X
Minv = np.linalg.inv(M)
diag = np.diag(Minv)
b=X.T@Y
beta = np.linalg.solve(M,b)
print("Coefficients beta")
print(beta)

# estimate variance in coefficients
Yhat = X@beta
delta=Y-Yhat
sigmaSq = (delta.T@delta)/(N-D-1)
print("Sigma^2")
print(sigmaSq)

# compute Z scores
z = beta/(np.sqrt(sigmaSq)*np.sqrt(diag))
print("Z score")
print(z)


X_eval=np.loadtxt("prostate_test.csv", skiprows=1, usecols=(1,2,3,4,5,6,7,8))

# evaluate model on test data
N_eval=X_eval.shape[0]
O_eval=np.ones((N_eval,1))
X_eval=np.concatenate((O_eval,X_eval),axis=1)
M_eval = X_eval.T@X_eval
Y_eval = X_eval@beta
print(Y_eval[:6])



X=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(1,2,3,4,5,6,8))
Y=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(9))
# fit linear regression by least squares
N=X.shape[0]
D=X.shape[1]
O=np.ones((N,1))
X=np.concatenate((O,X),axis=1)
M = X.T@X
Minv = np.linalg.inv(M)
diag = np.diag(Minv)
b=X.T@Y
beta = np.linalg.solve(M,b)
X_eval=np.loadtxt("prostate_test.csv", skiprows=1, usecols=(1,2,3,4,5,6,8))
# evaluate model on test data
N_eval=X_eval.shape[0]
O_eval=np.ones((N_eval,1))
X_eval=np.concatenate((O_eval,X_eval),axis=1)
M_eval = X_eval.T@X_eval
Y_eval = X_eval@beta
print(Y_eval[:6])

X=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(2,3,4,5,6,7,8))
Y=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(9))
# fit linear regression by least squares
N=X.shape[0]
D=X.shape[1]
O=np.ones((N,1))
X=np.concatenate((O,X),axis=1)
M = X.T@X
Minv = np.linalg.inv(M)
diag = np.diag(Minv)
b=X.T@Y
beta = np.linalg.solve(M,b)
X_eval=np.loadtxt("prostate_test.csv", skiprows=1, usecols=(2,3,4,5,6,7,8))
# evaluate model on test data
N_eval=X_eval.shape[0]
O_eval=np.ones((N_eval,1))
X_eval=np.concatenate((O_eval,X_eval),axis=1)
M_eval = X_eval.T@X_eval
Y_eval = X_eval@beta
print(Y_eval[:6])

