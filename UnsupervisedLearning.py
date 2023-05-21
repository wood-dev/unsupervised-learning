import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from Clustering import Clustering
from DimensionalityReduction import DimensionalityReduction
from util import getData, splitData
import numpy as np
from NeuralNetworks import NeuralNetworks

DATA_FOLDER = './data'

FILENAME_1 = 'online_shoppers_intention.csv'
CATEGORICAL_COLUMNS_1 = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
Y_COLUMN_1 = 'Revenue'
IDENTIFIER_1 = 1

FILENAME_2 = 'census_income.csv'
CATEGORICAL_COLUMNS_2 = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
Y_COLUMN_2 = 'greater-than-50k'
IDENTIFIER_2 = 2

def loadData_1(encode_category = False):
    fullFilename = os.path.join(DATA_FOLDER, FILENAME_1)
    df = pd.read_csv(fullFilename)
    df.head()

    global NUMERIC_COLUMNS
    NUMERIC_COLUMNS = df.columns.difference(CATEGORICAL_COLUMNS_1)
    NUMERIC_COLUMNS = NUMERIC_COLUMNS.drop(Y_COLUMN_1)

    if encode_category:
        df_oneHot = df[CATEGORICAL_COLUMNS_1]
        df_oneHot = pd.get_dummies(df_oneHot, drop_first=True)
        df_droppedOneHot = df.drop(CATEGORICAL_COLUMNS_1, axis=1)
        return pd.concat([df_oneHot, df_droppedOneHot], axis=1)
    else:
        return df

def loadData_2(encode_category = False):
    fullFilename = os.path.join(DATA_FOLDER, FILENAME_2)
    df = pd.read_csv(fullFilename)
    df.head()

    # categorical value
    df[CATEGORICAL_COLUMNS_2].fillna('Nan')
    # numeric value
    global NUMERIC_COLUMNS
    NUMERIC_COLUMNS = df.columns.difference(CATEGORICAL_COLUMNS_2)
    NUMERIC_COLUMNS = NUMERIC_COLUMNS.drop(Y_COLUMN_2)

    df_numeric = df[NUMERIC_COLUMNS]
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df_numeric = imputer.fit_transform(df_numeric)

    if encode_category:
        df_oneHot = df[CATEGORICAL_COLUMNS_2]
        df_oneHot = pd.get_dummies(df_oneHot, drop_first=True)
        df_droppedOneHot = df.drop(CATEGORICAL_COLUMNS_2, axis=1)
        return pd.concat([df_oneHot, df_droppedOneHot], axis=1)
    else:
        return df


def main():

    data1 = loadData_1(True)
    data2 = loadData_2(True)
    X1, y1 = getData(data1)
    X2, y2 = getData(data2)

    #### Part 1. Run the clustering algorithms on the datasets ####
    part1 = False
    findingOptimal = False
    if part1:
        print('---------- Part 1 --------')
        #### K-means clustering #####
        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        if findingOptimal:
            c.findOptimalClusterNumber()
        else:
            c.analyzeKmeanClustering(k=20, filenamePrefix='', feature1=16, feature2=22)
            c.analyzeEMClustering(k=20, filenamePrefix='', feature1=16, feature2=22)

        c = Clustering(IDENTIFIER_2, FILENAME_2, np.copy(X2), np.copy(y2))
        if findingOptimal:
            c.findOptimalClusterNumber()
        else:
            c.analyzeKmeanClustering(k=52, filenamePrefix='', feature1=16, feature2=22)
            c.analyzeEMClustering(k=52, filenamePrefix='', feature1=16, feature2=22)



    #### Part 2. Run the Dimensionality Reduction ####
    part2 = False
    findingOptimal = False
    if part2:
        print('---------- Part 2 --------')
        #### PCA ####
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, X1, y1)
        if findingOptimal:
            dr.runPCA()
        else:
            dr.analyzePCA(n_components=19, feature1=11, feature2=18)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, X2, y2)
        if findingOptimal:
            dr.runPCA()
        else:
            dr.analyzePCA(n_components=50, feature1=0, feature2=1)

        # #### ICA ####
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, X1, y1)
        if findingOptimal:
            dr.runICA()
        else:
            dr.analyzeICA(n_components=24, feature1=11, feature2=19)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, X2, y2)
        if findingOptimal:
            dr.runICA()
        else:
            dr.analyzeICA(n_components=78, feature1=0, feature2=1)

        # #### RP ####
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, X1, y1)
        if findingOptimal:
            dr.runRP()
        else:
            dr.analyzeRP(n_components=22, feature1=11, feature2=19)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, X2, y2)
        if findingOptimal:
            dr.runRP()
        else:
            dr.analyzeRP(n_components=78, feature1=0, feature2=1)

        # #### SVD ####
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, X1, y1)
        if findingOptimal:
            dr.RunTruncatedSVD()
        else:
            dr.analyzeSVD(n_components=16, feature1=11, feature2=15)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, X2, y2)
        if findingOptimal:
            dr.RunTruncatedSVD()
        else:
            dr.analyzeSVD(n_components=50, feature1=0, feature2=1)



    #### Part 3. Run the Dimensionality Reduction -> Clustering ####
    part3 = False
    findingOptimal = False
    if part3:

        print('---------- Part 3 --------')
        # DR PCA -> Clustering
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformPCA(19)
        filenamePrefix = 'DR-PCA-'
        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=16, filenamePrefix=filenamePrefix, feature1=16, feature2=18)
            c.analyzeEMClustering(k=16, filenamePrefix=filenamePrefix,feature1=16, feature2=18)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, np.copy(X2), np.copy(y2))
        X, y = dr.transformPCA(50)
        filenamePrefix = 'DR-PCA-'
        c = Clustering(IDENTIFIER_2, FILENAME_2, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=48, filenamePrefix=filenamePrefix,feature1=0, feature2=1)
            c.analyzeEMClustering(k=48, filenamePrefix=filenamePrefix,feature1=0, feature2=1)

        # DR ICA -> Clustering
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformICA(24)
        filenamePrefix = 'DR-ICA-'
        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=20, filenamePrefix=filenamePrefix,feature1=0, feature2=18)
            c.analyzeEMClustering(k=20, filenamePrefix=filenamePrefix,feature1=0, feature2=18)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, np.copy(X2), np.copy(y2))
        X, y = dr.transformICA(78)
        filenamePrefix = 'DR-ICA-'
        c = Clustering(IDENTIFIER_2, FILENAME_2, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=4, filenamePrefix=filenamePrefix,feature1=0, feature2=1)
            c.analyzeEMClustering(k=4, filenamePrefix=filenamePrefix,feature1=0, feature2=1)

        # DR RP -> Clustering
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformRP(22)
        filenamePrefix = 'DR-RP-'
        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=16, filenamePrefix=filenamePrefix,feature1=2, feature2=6)
            c.analyzeEMClustering(k=16, filenamePrefix=filenamePrefix,feature1=2, feature2=6)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, np.copy(X2), np.copy(y2))
        X, y = dr.transformRP(78)
        filenamePrefix = 'DR-RP-'
        c = Clustering(IDENTIFIER_2, FILENAME_2, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=76, filenamePrefix=filenamePrefix,feature1=0, feature2=1)
            c.analyzeEMClustering(k=76, filenamePrefix=filenamePrefix,feature1=0, feature2=1)

        # # DR SVD -> Clustering
        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformSVD(16)
        filenamePrefix = 'DR-SVD-'
        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=10, filenamePrefix=filenamePrefix,feature1=4, feature2=9)
            c.analyzeEMClustering(k=10, filenamePrefix=filenamePrefix,feature1=4, feature2=9)

        dr = DimensionalityReduction(IDENTIFIER_2, FILENAME_2, np.copy(X2), np.copy(y2))
        X, y = dr.transformSVD(50)
        filenamePrefix = 'DR-SVD-'
        c = Clustering(IDENTIFIER_2, FILENAME_2, np.copy(X), np.copy(y))
        if findingOptimal:
            c.findOptimalClusterNumber(filenamePrefix)
        else:
            c.analyzeKmeanClustering(k=44, filenamePrefix=filenamePrefix,feature1=0, feature2=1)
            c.analyzeEMClustering(k=44, filenamePrefix=filenamePrefix,feature1=0, feature2=1)

    part4 = False
    findingOptimal = False
    if part4:
        print('---------- Part 4 --------')

        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(np.copy(X1), np.copy(y1), 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        nn.analyzePerformance()

        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformPCA(19)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(5), max_iter=1000, solver='adam')


        X, y = dr.transformICA(24)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(15,15), max_iter=1000, solver='adam')

        X, y = dr.transformRP(22)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(20,20), max_iter=1000, solver='adam')

        X, y = dr.transformSVD(16)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(10), max_iter=1000, solver='adam')






    part5 = False
    findingOptimal = False
    if part5:
        print('---------- Part 5 --------')

        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformPCA(19)

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformKmeanClustering(k=16)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(5), max_iter=1000, solver='adam')

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformEMClustering(k=16)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(20,20), max_iter=1000, solver='adam')




        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformICA(24)

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformKmeanClustering(k=20)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(10,10,10,10), max_iter=1000, solver='adam')

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformEMClustering(k=20)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(5), max_iter=1000, solver='adam')


        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformRP(22)

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformKmeanClustering(k=16)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(25,25,25), max_iter=1000, solver='adam')

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformEMClustering(k=16)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(25,25), max_iter=1000, solver='adam')


        dr = DimensionalityReduction(IDENTIFIER_1, FILENAME_1, np.copy(X1), np.copy(y1))
        X, y = dr.transformSVD(16)

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformKmeanClustering(k=10)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(5), max_iter=1000, solver='adam')

        c = Clustering(IDENTIFIER_1, FILENAME_1, np.copy(X), np.copy(y))
        X_clustered = c.transformEMClustering(k=10)
        X_train_1, X_test_1, y_train_1, y_test_1 = splitData(X_clustered, y, 90)
        nn = NeuralNetworks(IDENTIFIER_1, FILENAME_1, X_train_1, X_test_1, y_train_1, y_test_1)
        if findingOptimal:
            nn.analyzeBestParameter()
        else:
            nn.analyzePerformance(hidden_layer_sizes=(5), max_iter=1000, solver='adam')




if __name__ == "__main__":
    main()