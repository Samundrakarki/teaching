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

X_train = np.linspace(-4,4,N)
X_train = X_train.reshape(-1,1)
	
y_train = np.zeros(N)
y_train = y_train.reshape(-1,1)

epsilon_train = np.random.normal(mu,sigma,N)
epsilon_train = epsilon_train.reshape(-1,1)

y_train = X_train*X_train + epsilon_train

kMax = 20

errors = np.zeros(kMax)
biasSqs = np.zeros(kMax)
noises = np.zeros(kMax)
variances = np.zeros(kMax)

for k in range(1,kMax+1):
	neigh = KNeighborsRegressor(n_neighbors=k)
	neigh.fit(X_train, y_train)
	
	x_0 = 1.12
	neighbors = neigh.kneighbors([[x_0]], return_distance=False)
	print((k,neighbors))
	
	E_f_hat = np.sum(X_train[neighbors]*X_train[neighbors])/k
	bias = x_0*x_0 - E_f_hat

	print((k,neighbors,E_f_hat,bias))
	
	variance = sigma*sigma/k
	
	error = sigma*sigma + bias*bias + variance

	errors[k-1]=error
	biasSqs[k-1]=bias*bias
	variances[k-1]=variance
	noises[k-1]=sigma*sigma

print(X_train)
plt.figure()
plt.plot(np.linspace(1,kMax,kMax),errors, label="error")
print(np.linspace(1,kMax,kMax))
print(errors)
print(biasSqs)
print(variances)
plt.plot(np.linspace(1,kMax,kMax),noises, label="noise")
plt.plot(np.linspace(1,kMax,kMax),biasSqs, label="squared bias")
plt.plot(np.linspace(1,kMax,kMax),variances, label="variance")
plt.legend()
#plt.plot(X_eval,X_eval*X_eval, label="exact model")
#plt.plot(X_train,y_train, 'rs', markersize=12, label="trainin set")
#plt.title("kNN regression for model $f(X)=X^2+\epsilon$ and $k=3$")
plt.show()


