import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt

directory = "data/"
data_file = "FinalCombined.csv"
iterations = 100000;
alpha = 0.0001;

def cleanData(data):
    y = data.iloc[:,-1]
    x = data
    x = x.iloc[:, :-1]
    row_count = len(x.index)
    initialized_rows = [1] * row_count
    x.insert(0, "Intercept", initialized_rows, True)
    return x, y

def main():
    data = pd.read_csv(directory + data_file)
    #trial = data[data["Joe Biden"]==1]
    #data = trial
    data = data.drop(columns=["Price"])
    x, y = cleanData(data)
    rows = len(x.index)
    columns = len(x.columns)
    x = x.to_numpy()
    #export_csv = y.to_csv(r'realvalues.csv', index = None, header=False)
    y = y.to_numpy()
    theta = np.zeros((1,columns))
    #theta, history = gradientDescent(x,y,theta,alpha,iterations)
    #dataframe = pd.DataFrame(history)
    #dataframe.plot()
    #plt.show()
    theta = normalEqn(x,y,theta)
    prediction = np.dot(x,np.transpose(theta))
    prediction = pd.DataFrame(prediction)
    initialized_rows = [0] * rows
    data.insert(len(data.columns), "ML MODEL", initialized_rows, True)
    data["ML MODEL"] = prediction
    data = data[data["Pete Buttigieg"]==1]
    export_csv = data.to_csv(r'prediction.csv', index = None, header=False)
    

def computeCost(x, y, theta):
    m = len(y)
    predictions = theta.dot(np.transpose(x))
    y = np.transpose(y)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost
    
def gradientDescent(x, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    y = np.matrix(y)
    y = np.transpose(y)
    for i in range (iterations):
        prediction = np.dot(x,np.transpose(theta))
        theta = theta - (1/m) * alpha * (np.transpose((prediction-y)).dot(x))
        cost_history[i] = computeCost(x,y,theta)
    return theta, cost_history
    
def normalEqn(x,y,theta):
    x_transpose = np.transpose(x)
    x_transpose_dot_x = x_transpose.dot(x)
    temp_1 = np.linalg.inv(x_transpose_dot_x)
    temp_2=x_transpose.dot(y)
    theta =temp_1.dot(temp_2)
    return theta
    
def neuralNetwork(x,y,theta):
    return 0  
main()