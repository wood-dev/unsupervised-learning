import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pandas as pd

GRAPH_FOLDER = './graph/'

def getFullFilePath(filename):
    return os.path.join(GRAPH_FOLDER, filename)

def splitData(X, Y, train_size):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=train_size/100)
    return X_train, X_test, y_train, y_test

def getData(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    scaler = MinMaxScaler(feature_range=[0,1])
    X_scaled = scaler.fit_transform(X)
    print('feature number: ', len(X_scaled[0]))
    return X_scaled, y

def saveGraph(plt, filename, identifier):
    plt.savefig(getFullFilePath(filename.format(id = identifier)), bbox_inches='tight')
    plt.close()