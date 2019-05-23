import matplotlib.pyplot as plt;
import numpy as np;


# Plot the Training Data
def plotData(X,y):
	plt.xlim(4,25);
	plt.ylim(-5,25);
	plt.title("Scatter plot of training data",color = "Blue",size = 15);
	plt.xlabel("Population of cities in 10,000s");
	plt.ylabel("Profit in $10,000s");
	plt.scatter(X,y,color="green",marker="x")
	