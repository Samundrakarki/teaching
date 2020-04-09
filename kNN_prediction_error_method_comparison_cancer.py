from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

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


N = 97

X=np.loadtxt("prostate.csv", skiprows=1, usecols=(1,2,3,4,5,6,7,8))
y=np.loadtxt("prostate.csv", skiprows=1, usecols=(9))

kMax = 40

plt.rcParams.update({'font.size': 16})
plt.figure()

for s in list([0]):

	indices = list(range(N));

	indices_perm = np.random.permutation(indices)

	errors = np.zeros(kMax)

	for k in range(1,kMax+1):

		curr_error = 0

		for l in range(N):		


			X_train = X[indices_perm]
			y_train = y[indices_perm]
			X_train = np.delete(X_train,[l],axis=0)
			y_train = np.delete(y_train,[l],axis=0)

			X_validate = X[indices_perm[l]]
			y_validate = y[indices_perm[l]]
			X_validate=[X_validate]
			y_validate=[y_validate]
			
			neigh = KNeighborsRegressor(n_neighbors=k)
			neigh.fit(X_train, y_train)
			y_prediction = neigh.predict(X_validate)
			curr_error = curr_error + (y_prediction-y_validate)**2
		
		errors[k-1] = curr_error/N

	plt.plot(np.linspace(1,kMax,kMax),errors,label="kNN reg. (leave-one-out CV)")
		

for s in list([0]):

	indices = list(range(N));

	indices_perm = np.random.permutation(indices)

	errors = np.zeros(kMax)

	for k in range(1,kMax+1):

		curr_error = 0


		X_train = X[indices_perm]
		y_train = y[indices_perm]

		neigh = KNeighborsRegressor(n_neighbors=k)
		neigh.fit(X_train, y_train)
		y_prediction = neigh.predict(X_train)
		curr_error = np.sum((y_prediction-y_train)**2)
		
		errors[k-1] = curr_error/N

	plt.plot(np.linspace(1,kMax,kMax),errors,label="kNN reg. (training error)")
		


curr_error = 0
for l in range(N):		

	X_train = X
	y_train = y
	X_train = np.delete(X_train,[l],axis=0)
	y_train = np.delete(y_train,[l],axis=0)

	X_validate = X[l]
	y_validate = y[l]
	X_validate = [X_validate]
	y_validate = [y_validate]

	lin_regr = linear_model.LinearRegression()
	lin_regr.fit(X_train, y_train)
	y_prediction = lin_regr.predict(X_validate)
	curr_error = curr_error + (y_prediction-y_validate)**2
		
error_lin = curr_error/N

plt.plot(np.linspace(1,kMax,kMax),error_lin*np.ones(kMax),label="linear reg. (leave-one-out CV)")


lin_regr = linear_model.LinearRegression()
lin_regr.fit(X, y)
y_prediction = lin_regr.predict(X)
lin_reg_training_error = sum((y_prediction-y)**2)/N

plt.plot(np.linspace(1,kMax,kMax),lin_reg_training_error*np.ones(kMax),label="linear reg. (training error)")



# for s in range(5):
# 
# 	indices = list(range(N));
# 
# 	indices_perm = np.random.permutation(indices)
# 
# 	K = 10
# 
# 	indices_perm = np.array_split(indices_perm,K);
# 	indices_perm=np.asarray(indices_perm)
# 	
# 	errors = np.zeros(kMax)
# 
# 	for k in range(1,kMax+1):
# 
# 		curr_error = 0
# 
# 		for l in range(K):		
# 
# 			subset_indices = list(range(K))
# 			curr_subset_index = l
# 			subset_indices.remove(curr_subset_index)
# 	
# 			indices_train = indices_perm[subset_indices]
# 			indices_train = np.concatenate(indices_train)
# 			indices_validate = indices_perm[curr_subset_index]
# 
# 			X_train = X[indices_train]
# 			y_train = y[indices_train]
# 
# 			X_validate = X[indices_validate]
# 			y_validate = y[indices_validate]
# 
# #			print(len(X_train))
# #			print(len(X_validate))
# 			
# 			neigh = KNeighborsRegressor(n_neighbors=k)
# 			neigh.fit(X_train, y_train)
# 			y_prediction = neigh.predict(X_validate)
# 			curr_error = curr_error + np.sum((y_prediction-y_validate)**2)/len(y_prediction)
# 		
# 		errors[k-1] = (curr_error/K)
# 
# 	if s>0:
# 		plt.plot(np.linspace(1,kMax,kMax),errors,color='orange', label="_nolegend_")
# 	else:
# 		plt.plot(np.linspace(1,kMax,kMax),errors,color='orange', label="10-fold CV")
# 




#print(np.linspace(1,kMax,kMax))
#print(errors)
#print(biasSqs)
#print(variances)
#plt.plot(np.linspace(1,kMax,kMax),biasSqs, label="squared bias")
#plt.plot(np.linspace(1,kMax,kMax),variances, label="variance")
plt.legend()
plt.xlim(kMax,1)
plt.xlabel("neighborhood size $k$ (in kNN)")
plt.ylabel("different errors using $L_2$ loss")
#plt.plot(X_eval,X_eval*X_eval, label="exact model")
#plt.plot(X_train,y_train, 'rs', markersize=12, label="trainin set")
plt.title("error analysis for prostate cancer data set")
plt.show()


