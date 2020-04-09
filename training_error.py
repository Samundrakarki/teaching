#%matplotlib notebook

from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(1,2,3,4,5,6,7,8))
y=np.loadtxt("prostate_training.csv", skiprows=1, usecols=(9))


errors_lin = np.zeros(67)
errors_neigh1 = np.zeros(67)
errors_neigh3 = np.zeros(67)

for N in range(67):
	print(N)

	lin_regr = linear_model.LinearRegression()
	lin_regr.fit(X[:N+1], y[:N+1])
	y_lin_pred = lin_regr.predict(X[:N+1])
	errors_lin[N] = np.sum((y_lin_pred-y[:N+1])*(y_lin_pred-y[:N+1]))/N
	
	neigh1 = KNeighborsRegressor(n_neighbors=1)
	neigh1.fit(X[:N+1], y[:N+1])
	y_neigh1_pred = neigh1.predict(X[:N+1])
	errors_neigh1[N] = np.sum((y_neigh1_pred-y[:N+1])*(y_neigh1_pred-y[:N+1]))/N
	
	if N>2:
		neigh3 = KNeighborsRegressor(n_neighbors=3)
		neigh3.fit(X[:N+1], y[:N+1])
		y_neigh3_pred = neigh3.predict(X[:N+1])
		errors_neigh3[N] = np.sum((y_neigh3_pred-y[:N+1])*(y_neigh3_pred-y[:N+1]))/N


plt.rcParams.update({'font.size': 18})
plt.figure()
plt.plot(np.linspace(1,N+1,N+1),errors_lin, label="linear regression")
plt.plot(np.linspace(1,N+1,N+1),errors_neigh1, label="kNN regression $(k=1)$")
plt.plot(np.linspace(1,N+1,N+1),errors_neigh3, label="kNN regression $(k=3)$")
plt.xlabel("N (no. of training samples)")
plt.ylabel("training error")
#plt.plot(np.linspace(1,kMax,kMax),noises, label="noise")
#plt.plot(np.linspace(1,kMax,kMax),biasSqs, label="squared bias")
#plt.plot(np.linspace(1,kMax,kMax),variances, label="variance")
plt.legend()
plt.show()


errors_neigh = np.zeros(67)

for k in range(67):
	neigh = KNeighborsRegressor(n_neighbors=k+1)
	neigh.fit(X, y)
	y_neigh_pred = neigh.predict(X)
	errors_neigh[k] = np.sum((y_neigh_pred-y)*(y_neigh_pred-y))/N
	
plt.rcParams.update({'font.size': 18})
plt.figure()
plt.xlim(67,1)
plt.plot(np.linspace(1,N+1,N+1),errors_neigh, label="kNN regression")
plt.xlabel("k (neighborhood size in kNN regression)")
plt.ylabel("training error")
plt.legend()
plt.show()

