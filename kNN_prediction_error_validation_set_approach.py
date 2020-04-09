from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

#N = 20
#mu = 0
#sigma = 0.6
#k=3
#
#X_train = np.linspace(-4,4,N)
#X_train = X_train.reshape(-1,1)
#
#y_train = np.zeros(N)
#y_train = y_train.reshape(-1,1)
#
#epsilon_train = np.random.normal(mu,sigma,N)
#epsilon_train = epsilon_train.reshape(-1,1)
#
#y_train = X_train*X_train + epsilon_train
#
#
#neigh = KNeighborsRegressor(n_neighbors=k)
#neigh.fit(X_train, y_train)
#
#X_eval = np.linspace(-4,4,1000)
#X_eval = X_eval.reshape(-1,1)
#
#plt.figure()
#plt.plot(X_eval,neigh.predict(X_eval), label="kNN regression predictor")
#plt.plot(X_eval,X_eval*X_eval, label="exact model")
#plt.plot(X_train,y_train, 'rs', markersize=12, label="trainin set")
#plt.title("kNN regression for model $f(X)=X^2+\epsilon$ and $k=3$")
#plt.show()



mu = 0.0
sigma = 1.15
N= 51
N_train = 40

X = np.linspace(-4,4,N)
X = X.reshape(-1,1)
	
y = np.zeros(N)
y = y.reshape(-1,1)

epsilon = np.random.normal(mu,sigma,N)
epsilon = epsilon.reshape(-1,1)

y = X*X + epsilon

kMax = 20

plt.rcParams.update({'font.size': 16})
plt.figure()

for s in range(5):

	indices = list(range(N));

	indices_perm = np.random.permutation(indices)

	X_train = X[indices_perm[:N_train]]
	y_train = y[indices_perm[:N_train]]
	X_validate = X[indices_perm[N_train:]]
	y_validate = y[indices_perm[N_train:]]
	
	errors = np.zeros(kMax)

	for k in range(1,kMax+1):
		neigh = KNeighborsRegressor(n_neighbors=k)
		neigh.fit(X_train, y_train)
		y_prediction = neigh.predict(X_validate)
		errors[k-1] = np.sum((y_prediction-y_validate)**2)/len(y_prediction)

	plt.plot(np.linspace(1,kMax,kMax),errors)

#print(np.linspace(1,kMax,kMax))
#print(errors)
#print(biasSqs)
#print(variances)
#plt.plot(np.linspace(1,kMax,kMax),biasSqs, label="squared bias")
#plt.plot(np.linspace(1,kMax,kMax),variances, label="variance")
plt.legend()
plt.xlabel("neighborhood size $k$")
plt.ylabel("error computed by validation set approach using $L_2$ loss")
#plt.plot(X_eval,X_eval*X_eval, label="exact model")
#plt.plot(X_train,y_train, 'rs', markersize=12, label="trainin set")
plt.title("influence of different splits on error prediction (kNN regression error)")
plt.show()


