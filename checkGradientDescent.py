import matplotlib.pyplot as plt;
import numpy as np;



""" Plot the values of cost vs the number of iterations in
		Gradient Descent """
def plotCost_vs_iter(cost_values_history,iterations):
	turns = np.arange(1,iterations+1,1);
	plt.plot(turns, cost_values_history, color = "green", label = "line");
	plt.xlim(0,iterations);
	plt.ylim(np.min(cost_values_history)-1,np.max(cost_values_history)+1);
	plt.xlabel("Number of Iterations");
	plt.ylabel("Cost found using Gradient Descent");
	plt.title("Variation of Cost function with \nnumber of iterations of Gradient Descent" , color = "Blue" ,size = 15)
	plt.legend();
	plt.grid(linestyle = ":");