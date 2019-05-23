import matplotlib.pyplot as plt;
import numpy as np;

# Plot the Training Data
def plotData(X,y):
	plt.scatter(X,y,color="green",marker="x");
	plt.xlim(4,25);
	plt.ylim(-5,25);
	plt.title("Scatter plot of training data",color = "Blue",size = 20);
	plt.xlabel("Population of cities in 10,000s");
	plt.ylabel("Profit in $10,000s");
	plt.show(block = False);
	plt.pause(2);
	plt.close();

def setParams(X):
	# Adding the column of ones to X
	X = np.c_[np.ones(np.size(X)) , X];
	theta = np.zeros((2,1));
	# print(theta);
	iterations = 1500;
	alpha = 0.01;
	return X,theta,iterations,alpha;

# Compute the cost function
def costFunction(X,y,theta):
	m = np.size(y);
	J = 0;
	for i in range(X.shape[0]):
		J += (np.dot(X[i,:],theta) - y[i]) ** 2;
	J /= 2*m;
	return J;


# open data file
path = "./data_files/data1.txt";
X , y = np.genfromtxt(path,delimiter = ",", unpack = True);


# Function call to plot the training data as a scatter plot
plotData(X,y);

# Setting up the parameters
X , theta , iterations , alpha = setParams(X);


cost = costFunction(X,y,theta);
print(cost);

