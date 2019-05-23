import matplotlib.pyplot as plt;
import numpy as np;



# Compute the cost function
def costFunction(X,y,theta):
	m = y.shape[0];
	J = 0;
	for i in range(X.shape[0]):
		J += (np.dot(X[i,:],theta) - y[i]) ** 2;
	J /= 2*m;
	return J;