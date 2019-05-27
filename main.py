import matplotlib.pyplot as plt;
import numpy as np;

from gradientDescent import gradientDescent;
from setParams import setParams;
from costFunction import costFunction;
from checkGradientDescent import plotCost_vs_iter;
from plotData import plotData;




# open data file
print("Opening Data file...");
path = "./data_files/data1.txt";
X , y = np.genfromtxt(path,delimiter = ",", unpack = True);


# Function call to plot the training data as a scatter plot
print ("\n\n\nPlotting data point for visualization...");
plotData(X,y);
print ("Press any key to continue...");
plt.waitforbuttonpress();
plt.close();

# Setting up the parameters
print ("\n\n\nSetting up the parameters...\n"
		"Intercept term add to population data\n"
		"Max iteration chosen to be 1500\n"
		"Learning rate set to 0.01");

mod_X,theta , iterations , alpha = setParams(X);


# Gradient Descent
cost_values_history , theta = gradientDescent(mod_X,y,theta,iterations,alpha);
print("The optimum values of parameters for the linear fit was found to be : " + str(theta[0][0]) + " , " + str(theta[1][0]));
print("\nThe minimum cost found using Gradient Descent algorithm is :" + " " + str(cost_values_history[-1]));
plotCost_vs_iter(cost_values_history,iterations);
plt.show(block = False);
print ("Press any key to continue...");
plt.waitforbuttonpress();
plt.close();



# Plotting the best fit straight line to the data found using linear regression algorithm
print ("\n\n\nPlotting the best fit straight line to the data found using linear regression algorithm");
plt.scatter(X,y,color="green",marker="x")
xaxis = np.array([5,20]);
yaxis = np.array([float(theta[0][0]+5*theta[1][0]), float(theta[0][0]+20*theta[1][0])]);
plt.plot(xaxis,yaxis, label = "Prediction line", linewidth=2)
plt.xlim(4,25);
plt.ylim(-5,25);
plt.title("Best fit straight line",color = "k",size = 15);
plt.xlabel("Population of cities in 10,000s");
plt.ylabel("Profit in $10,000s");
plt.legend();
plt.grid(linestyle = ":");
plt.show(block = False);
print ("Press any key to continue...");
plt.waitforbuttonpress();
print("Thank you!!\n");
plt.close();