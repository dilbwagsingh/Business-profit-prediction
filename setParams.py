import matplotlib.pyplot as plt;
import numpy as np;



def setParams(X):
	# Adding the column of ones to X
	X = np.c_[np.ones(np.size(X)) , X];
	theta = np.zeros((2,1));
	# print(theta);
	iterations = 1500;
	alpha = 0.01;
	return X,theta,iterations,alpha;