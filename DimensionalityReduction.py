import matplotlib.pyplot as plt
from util import splitData, getFullFilePath, saveGraph
from time import time
import numpy as np
import matplotlib.cm as cm
import warnings
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from scipy.stats import kurtosis
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import euclidean_distances


warnings.filterwarnings("ignore")


RANDOM_SEED = 100

class DimensionalityReduction:

    def __init__(self):
        pass

    def __init__(self, id, title, X, y):
        self.identifier = id
        self.title = title
        self.X = X
        self.y = y


    def transformPCA(self, n_components):
        print('PCA Transform on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_r = pca.fit_transform(X)

        # scaler = MinMaxScaler(feature_range=[0,1])
        # X_r = scaler.fit_transform(X_r)
        return X_r, y


    def runPCA(self):

        print('Plotting PCA Chart on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y

        pca = PCA(random_state=RANDOM_SEED)
        X_r = pca.fit_transform(X)
        #print('explained variance ratio: %s' % str(pca.explained_variance_ratio_))

        plt.figure()
        colors = ["b","g"]

        for color, i in zip(colors, [0,1]):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, label=i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title("PCA of {title}".format(title=self.title))
        filename = "PCA-{id}.png"
        saveGraph(plt, filename, self.identifier)

        eigenvalues = pca.singular_values_
        ratios = pca.explained_variance_ratio_

        fig, ax = plt.subplots()
        plt.plot(np.arange(1,len(eigenvalues)+1,1) , eigenvalues , 'o')
        plt.plot(np.arange(1,len(eigenvalues)+1,1) , eigenvalues , '-', alpha = 0.8)
        plt.xlabel('Component number') , plt.ylabel('Eigenvalue')
        plt.title("PCA Eigenvalues of {title}".format(title=self.title))
        filename = "PCA-Eigenvalues-{id}.png"
        saveGraph(plt, filename, self.identifier)

        ratios = np.cumsum(ratios)
        fig, ax = plt.subplots()
        plt.plot(np.arange(1,len(ratios)+1,1) , ratios , 'o' )
        plt.plot(np.arange(1,len(ratios)+1,1) , ratios , '-', alpha = 0.8)
        plt.axhline(y=0.9, color='g', linestyle='--')
        plt.xlabel('Component number') , plt.ylabel('Cummulative variance')
        plt.title("PCA Cummulative Variance of {title}".format(title=self.title))
        filename = "PCA-Variance-{id}.png"
        saveGraph(plt, filename, self.identifier)
        print('variance above 0.99: ')
        print(*(np.argwhere(ratios > max(ratios) * 0.99)), sep=', ')

    def analyzePCA(self, n_components, feature1=0, feature2=1):

        print('Analyzing PCA with {n_components} compoents on dataset {title}...'.format(n_components=n_components,title=self.title))
        X = self.X
        y = self.y
        time1 = time()
        pca = PCA(n_components = n_components, random_state=RANDOM_SEED)
        X_r = pca.fit_transform(X)
        print('runtime: ', time()-time1)
        plt.figure()
        colors = ["b","g"]
        for color, i in zip(colors, [0,1]):
            plt.scatter(X_r[y == i, feature1], X_r[y == i, feature2], color=color, alpha=.8, label=i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel("Feature space for the feature %d " % feature1)
        plt.ylabel("Feature space for the feature %d " % feature2)
        plt.title('PCA of {title} with n_components = {n_components}'.format(title=self.title, n_components=n_components))
        filename = "PCA-{id}-{n_components}.png".format(id=self.identifier,n_components=n_components)
        saveGraph(plt, filename, self.identifier)

    def transformICA(self, n_components):
        print('ICA Transform on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y
        ica = FastICA(n_components=n_components, random_state=RANDOM_SEED)
        X_r = ica.fit_transform(X)
        return X_r, y

    def runICA(self):

        print('Running ICA on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y

        kurto = []
        list_error = []
        _, n_features = X.shape
        range_component_number = np.arange(2, n_features, 2)

        for component_number in range_component_number:
            transformer = FastICA(n_components=component_number, random_state=RANDOM_SEED, tol=0.001)
            X_r = transformer.fit_transform(X)
            list_error.append(self.reconstructionError(transformer, X))
            kurt = kurtosis(X_r, axis=0)
            kurt = np.mean(np.abs(kurt))
            kurto.append(kurt)
            #print('n_clusters = ', n_clusters, ' Kurtosis : ', kurt)

        fig, ax = plt.subplots()
        plt.plot(range_component_number, kurto , 'o')
        plt.plot(range_component_number, kurto , '-', alpha = 0.8)
        plt.axhline(y=max(kurto)*0.9, color='g', linestyle='--')
        plt.title('ICA Kurtosis on {title}'.format(title=self.title))
        plt.xlabel('Number of components') , plt.ylabel('Average kurtosis')
        filename = 'ICA-kurtosis-{id}.png'.format(id=self.identifier)
        saveGraph(plt, filename, self.identifier)

        print('error below 0.005: ')
        print(*(np.argwhere(np.array(list_error) < 0.005)) *2, sep=', ')

        fig, ax = plt.subplots()
        plt.plot(range_component_number , list_error , 'o', markersize = 2 )
        plt.plot(range_component_number, list_error, '-', alpha = 0.5)
        plt.axhline(y=max(list_error)*0.1, color='g', linestyle='--')
        plt.title("ICA - Reconstruction error of {title}".format(title=self.title))
        plt.xlabel('Number of components') , plt.ylabel('Reconstruction error')
        filename = 'ICA-Error-{id}.png'.format(id=self.identifier)
        saveGraph(plt, filename, self.identifier)

    def analyzeICA(self, n_components, feature1=0, feature2=1):

        print('Analyzing ICA with {n_components} compoents on dataset {title}...'.format(n_components=n_components,title=self.title))
        X = self.X
        y = self.y
        time1 = time()
        ica = FastICA(n_components=n_components,random_state=RANDOM_SEED)
        X_r = ica.fit_transform(X)
        print('runtime: ', time()-time1)

        plt.figure()
        colors = ["b","g"]
        for color, i in zip(colors, [0,1]):
            plt.scatter(X_r[y == i, feature1], X_r[y == i, feature2], color=color, alpha=.8, label=i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel("Feature space for the feature %d " % feature1)
        plt.ylabel("Feature space for the feature %d " % feature2)
        plt.title("ICA of {title}".format(title=self.title))
        filename = "ICA-{id}.png"
        saveGraph(plt, filename, self.identifier)


    def reconstructionError(self,projections,X):
        W = projections.components_
        if sps.issparse(W):
            W = W.todense()
        p = pinv(W)
        reconstructed = ((p@W)@(X.T)).T # Unproject projected data
        errors = np.square(X-reconstructed)
        return np.nanmean(errors)

    def transformRP(self, n_components):
        print('RP Transform on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y
        srp = SparseRandomProjection(n_components=n_components, random_state=RANDOM_SEED)
        X_r = srp.fit_transform(X)
        return X_r, y

    def runRP(self):

        print('Running RP on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y

        _, n_featuers = X.shape
        range_component_number = np.arange(2,n_featuers,2)
        list_error_avg = [] ; list_kurtosis_avg = []
        for i in range(0,10):
            list_error = [] ; list_kurtosis = []
            for n_components in range_component_number:
                grp = SparseRandomProjection(n_components=n_components,  random_state=np.random.randint(1000))
                X_r = grp.fit_transform(X)
                list_error.append(self.reconstructionError(grp, X))

                kurt = kurtosis(X_r, axis=0)
                kurt = np.mean(np.abs(kurt))
                list_kurtosis.append(kurt)

            list_kurtosis_avg.append(list_kurtosis)
            list_error_avg.append(list_error)

        list_error_avg = np.array(list_error_avg)
        list_kurtosis_avg = np.array(list_kurtosis_avg)

        print('error below 0.005: ')
        print(*(np.argwhere(np.array(list_error) < 0.005))*2, sep=', ')
        rec_mean = np.mean(list_error_avg, axis = 0)
        rec_std  = np.std(list_error_avg, axis = 0)

        kurtosis_mean = np.mean(list_kurtosis_avg, axis = 0)
        kurtosis_std  = np.std(list_kurtosis_avg, axis = 0)

        fig, ax = plt.subplots()
        plt.plot(range_component_number , rec_mean , 'o', markersize = 2 )
        plt.plot(range_component_number, rec_mean, '-', alpha = 0.5)
        plt.axhline(y=max(rec_mean)*0.1, color='g', linestyle='--')
        plt.fill_between(range_component_number, rec_mean - rec_std,
                        rec_mean + rec_std, alpha=0.2)

        plt.title("Randomized projections - Reconstruction error of {title}".format(title=self.title))
        plt.xlabel('Number of components') , plt.ylabel('Reconstruction error')

        filename = 'RP-Error-{id}.png'.format(id=self.identifier)
        saveGraph(plt, filename, self.identifier)

        fig, ax = plt.subplots()
        plt.plot(range_component_number , kurtosis_mean , 'o', markersize = 2 )
        plt.plot(range_component_number, kurtosis_mean, '-', alpha = 0.5)
        plt.axhline(y=max(kurtosis_mean)*0.1, color='g', linestyle='--')
        plt.fill_between(range_component_number, kurtosis_mean - kurtosis_std,
                        kurtosis_mean + kurtosis_std, alpha=0.2)

        plt.title("Randomized projections - Kurtosis of {title}".format(title=self.title))
        plt.xlabel('Number of components') , plt.ylabel('Kurtosis')

        filename = 'RP-Kurtosis-{id}.png'.format(id=self.identifier)
        saveGraph(plt, filename, self.identifier)


    def analyzeRP(self, n_components, feature1=0, feature2=1):
        print('Analyzing RP with {n_components} compoents on dataset {title}...'.format(n_components=n_components,title=self.title))
        X = self.X
        y = self.y
        time1 = time()
        srp = SparseRandomProjection(n_components=n_components, random_state=RANDOM_SEED)
        X_r = srp.fit_transform(X)
        print('runtime: ', time()-time1)
        plt.figure()
        colors = ["b","g"]
        lw = 2
        for color, i in zip(colors, [0, 1]):
            plt.scatter(X_r[y == i, feature1], X_r[y == i, feature2], color=color, alpha=.8, label=i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel("Feature space for the feature %d " % feature1)
        plt.ylabel("Feature space for the feature %d " % feature2)
        plt.title('Random Projection of {title} with n_components = {n_components}'.format(title=self.title, n_components=n_components))
        filename = "RP-{id}-{n_components}.png".format(id=self.identifier,n_components=n_components)
        saveGraph(plt, filename, self.identifier)


    def transformSVD(self, n_components):
        print('SVD Transform on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        X_r = svd.fit_transform(X)
        return X_r, y


    def analyzeSVD(self, n_components,feature1=0,feature2=1):
        print('Analyzing SVD with {n_components} compoents on dataset {title}...'.format(n_components=n_components,title=self.title))
        X = self.X
        y = self.y
        time1 = time()
        transformer = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        X_r = transformer.fit_transform(X)
        print('runtime: ', time()-time1)
        plt.figure()
        colors = ["b","g"]
        lw = 2
        for color, i in zip(colors, [0, 1]):
            plt.scatter(X_r[y == i, feature1], X_r[y == i, feature2], color=color, alpha=.8, label=i)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel("Feature space for the feature %d " % feature1)
        plt.ylabel("Feature space for the feature %d " % feature2)
        plt.title('Truncated SVD of {title} with n_components = {n_components}'.format(title=self.title, n_components=n_components))
        filename = "SVD-{id}-{n_components}.png".format(id=self.identifier,n_components=n_components)
        saveGraph(plt, filename, self.identifier)

    def RunTruncatedSVD(self):

        print('Running Truncated SVD on dataset {title}...'.format(title=self.title))
        X = self.X
        y = self.y

        _, n_features = X.shape
        range_component_number = np.arange(2,n_features,2)

        list_error = []
        for n_components in range_component_number:
            transformer = TruncatedSVD(n_components=n_components,  random_state=RANDOM_SEED)
            X_r = transformer.fit_transform(X)
            list_error.append(self.reconstructionError(transformer, X))

        print('error below 0.005: ')
        print(*(np.argwhere(np.array(list_error) < 0.005))*2, sep=', ')

        fig, ax = plt.subplots()
        plt.plot(range_component_number , list_error , 'o', markersize = 2 )
        plt.plot(range_component_number, list_error, '-', alpha = 0.5)
        plt.axhline(y=max(list_error)*0.1, color='g', linestyle='--')
        plt.title("Truncated SVD - Reconstruction error of {title}".format(title=self.title))
        plt.xlabel('Number of components') , plt.ylabel('Reconstruction error')
        filename = 'SVD-Error-{id}.png'.format(id=self.identifier)
        saveGraph(plt, filename, self.identifier)
