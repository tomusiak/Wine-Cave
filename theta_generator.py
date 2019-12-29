import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import preprocessing

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense

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
    data = data[data["Joe Biden"]==1]
    #data = data.drop(columns=["Price"])
    x, y = cleanData(data)
    x = pd.DataFrame(x)
    x = x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    num_rows = len(x.index)
    num_columns = len(x.columns)
    x = x.to_numpy()
    y = y.to_numpy()
    theta = np.zeros((1,num_columns))
    #theta = gradientDescent(x,y,theta,alpha,iterations)
    #theta = normalEqn(x,y,theta)
    neuralNetwork(x,y);
    prediction = np.dot(x,np.transpose(theta))
    prediction = pd.DataFrame(prediction)
    initialized_rows = [0] * num_rows
    data.insert(len(data.columns), "Predicted Values", initialized_rows, True)
    data["Predicted Values"] = prediction
    data = data[data["Joe Biden"]==1]
    export_csv = data.to_csv(r'prediction.csv', index = None, header=True)
    
    

def computeCost(x, y, theta):
    m = len(y)
    predictions = theta.dot(np.transpose(x))
    y = np.transpose(y)
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
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
    dataframe = pd.DataFrame(history)
    dataframe.plot()
    plt.show()
    return theta
    
def normalEqn(x,y,theta):
    x_transpose = np.transpose(x)
    x_transpose_dot_x = x_transpose.dot(x)
    temp_1 = np.linalg.inv(x_transpose_dot_x)
    temp_2=x_transpose.dot(y)
    theta =temp_1.dot(temp_2)
    return theta

def neuralNetwork(x,y):
    x = pd.DataFrame(x)
x = x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)
    model = Sequential()
    model.add(Dense(500, input_dim=13, activation= "relu"))
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(50, activation= "relu"))
    model.add(Dense(1))
    model.compile(loss= "mae" , optimizer="adam", metrics=["mae"])
    model.fit(X_train, y_train, epochs=200)
    pred_train= model.predict(X_train)
    pred= model.predict(X_test)
    pred = pd.DataFrame(pred)
    y_test = pd.DataFrame(y_test)
    y_test.to_csv(r'testings.csv',index=None,header=False)
    pred.to_csv(r'idk.csv',index=None,header = False)

main()