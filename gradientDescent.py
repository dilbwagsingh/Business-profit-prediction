import matplotlib.pyplot as plt;
import numpy as np;


from costFunction import costFunction;


# Compute optimum values of theta using Gradient Descent
def gradientDescent(X,y,theta,iterations,alpha):
	m = y.shape[0];
	J_history = np.zeros((iterations));
	for turn in range(iterations):
		h0 = 0;
		h1 = 0;
		for i in range(m):
			h0 += (np.dot(X[i,0:2],theta) - y[i]); 
			h1 += (np.dot(X[i,0:2],theta) - y[i])*X[i,1];
		temp0 = theta[0] - (alpha/m)*h0;
		temp1 = theta[1] - (alpha/m)*h1;
		theta[0] = temp0;
		theta[1] = temp1;
		J_history[turn] = costFunction(X,y,theta);
	return J_history,theta;
