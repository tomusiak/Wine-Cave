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
iterations = 10000;
alpha = 0.01;

def addIntercept(x):
    row_count = len(x.index)
    initialized_rows = [1] * row_count
    x.insert(0, "Intercept", initialized_rows, True)
    x = pd.DataFrame(x)
    return x

def dropFeatures(data, featureList):
    data = data.drop(columns=featureList)
    return data

def cleanData(data):
    y = data.iloc[:,-1]
    x = data.iloc[:, :-1]
    x = pd.DataFrame(x)
    x = x.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x)
    return x, y

def splitData(x,y,proportion):
    row_count = len(x.index)
    y = pd.DataFrame(y)
    x = pd.DataFrame(x)
    data = x
    data['y'] = y
    shuffledData= data.sample(frac=1)
    split = round(row_count*proportion)
    testData = shuffledData.head(split)
    trainingData = shuffledData.tail(row_count-split)
    y_train = pd.DataFrame(trainingData.iloc[:,-1])
    x_train = pd.DataFrame(trainingData.iloc[:, :-1])
    y_test = pd.DataFrame(testData.iloc[:,-1])
    x_test = pd.DataFrame(testData.iloc[:,:-1])
    return x_train, y_train, x_test, y_test

def main():
    data = pd.read_csv(directory + data_file)
    data = dropFeatures(data,["Price"])
    x, y = cleanData(data)
    x_train, y_train, x_test, y_test = splitData(x,y,proportion = .20)
    print("Starting")
    gradient_descent_predictions = gradientDescentPrediction(x_train,y_train,x_test,y_test)
    print("Gradient Descent Done")
    #normal_equation_predictions = normalEquationPrediction(x_train,y_train,x_test,y_test)
    print("Normal Equation Done")
    #regression_nn_predictions = regressionNNPrediction(x_train,y_train, x_test, y_test)
    print("Regression NN done")
    
def gradientDescentPrediction(x_train,y_train, x_test, y_test):
    x_train = addIntercept(x_train)
    x_test = addIntercept(x_test)
    theta = gradientDescent(x_train,y_train,alpha,iterations)
    prediction = np.dot(x_test,np.transpose(theta))
    prediction = pd.DataFrame(prediction)
    print(x_train.dtypes)
    print(y_train.dtypes)
    print(x_test.dtypes)
    print(y_test.dtypes)
    print(prediction.dtypes)
    return prediction

def gradientDescent(x, y, alpha, iterations):
    theta = np.zeros((1,len(x.columns)))
    m = len(y)
    cost_history = np.zeros(iterations)
    y = np.matrix(y)
    for i in range (iterations):
        predictions = np.dot(x,np.transpose(theta))
        theta = theta - (1/m) * alpha * (np.transpose((predictions-y)).dot(x))
        cost_history[i] = computeCost(predictions, y, m)
    dataframe = pd.DataFrame(cost_history)
    dataframe.plot()
    plt.show()
    theta = pd.DataFrame(theta)
    return theta

def computeCost(predictions, y, m):
    cost = (1/(2*m)) * np.sum(np.square(predictions-y))
    return cost

def normalEquationPrediction(x_train, y_train, x_test, y_test):
    theta = np.zeros((1,len(x_train.columns)))
    theta = normalEqn(x_train,y_train)
    prediction = np.dot(x_test,np.transpose(theta))
    return prediction
    
def normalEqn(x,y):
    x = addIntercept(x)
    x_transpose = np.transpose(x)
    x_transpose_dot_x = x_transpose.dot(x)
    temp_1 = np.linalg.inv(x_transpose_dot_x)
    temp_2=x_transpose.dot(y)
    theta =temp_1.dot(temp_2)
    return theta

def regressionNNPrediction(x_train, y_train, x_test, y_test):
    prediction = regression_NN(x_train, y_train, x_test)
    return prediction

def regression_NN(x_train, y_train, x_test):
    model = Sequential()
    model.add(Dense(500, input_dim=11, activation= "relu"))
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(50, activation= "relu"))
    model.add(Dense(1))
    model.compile(loss= "mae" , optimizer="adam", metrics=["mae"])
    model.fit(x_train, y_train, epochs=100)
    pred = model.predict(x_test)
    return pred
        
main()