import matplotlib.pyplot as plt;
import numpy as np;





# Plot the Training Data
def plotData(X,y):
	plt.scatter(X,y,color="green",marker="x");
	plt.xlim(4,25);
	plt.ylim(-5,25);
	plt.title("Scatter plot of training data",color = "Blue",size = 15);
	plt.xlabel("Population of cities in 10,000s");
	plt.ylabel("Profit in $10,000s");
	plt.show(block = False);
	plt.pause(2);
	plt.close();

""" Plot the values of cost vs the number of iterations in
		Gradient Descent """
def plotCost_vs_iter(cost_values_history,iterations):
	turns = np.arange(1,iterations+1,1);
	plt.plot(turns, cost_values_history, color = "green");
	plt.xlim(0,iterations);
	plt.ylim(np.min(cost_values_history)-1,np.max(cost_values_history)+1);
	plt.xlabel("Number of Iterations");
	plt.ylabel("Cost found using Gradient Descent");
	plt.title("Variation of Cost function with \n number of iterations of Gradient Descent" , color = "Blue" ,size = 15)
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
	m = y.shape[0];
	J = 0;
	for i in range(X.shape[0]):
		J += (np.dot(X[i,:],theta) - y[i]) ** 2;
	J /= 2*m;
	return J;

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







# open data file
path = "./data_files/data1.txt";
X , y = np.genfromtxt(path,delimiter = ",", unpack = True);

# Function call to plot the training data as a scatter plot
plotData(X,y);

# Setting up the parameters
X , theta , iterations , alpha = setParams(X);


cost = costFunction(X,y,theta);
# print(cost);


# Gradient Descent
cost_values_history , theta = gradientDescent(X,y,theta,iterations,alpha);
plotCost_vs_iter(cost_values_history,iterations);

