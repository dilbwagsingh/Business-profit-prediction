import matplotlib.pyplot as plt;
import numpy as np;

from gradientDescent import gradientDescent;
from setParams import setParams;
from costFunction import costFunction;
from checkGradientDescent import plotCost_vs_iter;
from plotData import plotData;




# open data file
path = "./data_files/data1.txt";
X , y = np.genfromtxt(path,delimiter = ",", unpack = True);

# Function call to plot the training data as a scatter plot
plotData(X,y);
plt.pause(2);
plt.close();

# Setting up the parameters
mod_X,theta , iterations , alpha = setParams(X);


cost = costFunction(mod_X,y,theta);
# print(cost);


# Gradient Descent
cost_values_history , theta = gradientDescent(mod_X,y,theta,iterations,alpha);
plotCost_vs_iter(cost_values_history,iterations);
plt.show(block = False);
plt.pause(2);
plt.close();



# plt.xlim(4,25);
# plt.ylim(-5,25);
# plt.title("Scatter plot of training data",color = "Blue",size = 15);
# plt.xlabel("Population of cities in 10,000s");
# plt.ylabel("Profit in $10,000s");
# #plt.scatter(X,y,color="green",marker="x")
# xaxis = np.array([0,1]);
# yaxis = np.array([float(theta[0][0]), float(theta[0][0]+theta[1][0])]);
# plt.plot(xaxis,yaxis, marker = "o", color = "blue")
# plt.show();
# plt.pause(2);
# plt.close();