import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, x, y, cv=kfold)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print(results)
    

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

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

main()