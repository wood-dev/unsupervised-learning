import matplotlib.pyplot as plt
from util import splitData, getFullFilePath, saveGraph
from sklearn.cluster import KMeans
from time import time
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")


RANDOM_SEED = 100

class Clustering:


    def __init__(self, id, title, X, y):

        self.identifier = id
        self.title = title
        self.X = X
        self.y = y

    def findOptimalClusterNumber(self, filenamePrefix=''):

        print('Finding best k cluster number...')

        X = self.X

        kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": RANDOM_SEED,
        }

        k_range = range(2, len(X[0]), 2)
        k_range_xtick = range(2, len(X[0]), 4)

        # method 1: Knee -------------
        sse = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)

        kl = KneeLocator(
            k_range, sse, curve="convex", direction="decreasing"
        )
        print('kl.elbow: ', kl.elbow)

        plt.plot(k_range, sse)
        plt.xticks(k_range_xtick)
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        filename = filenamePrefix + 'Clustering-FindK-Kneed-{id}.png'
        saveGraph(plt, filename, self.identifier)

        # method 2: silhouette coefficient ------------
        silhouette_coefficients = []
        silhouette_values = []
        davies_bouldin_score_list = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(X)
            silhouette_coefficients.append(silhouette_score(X, kmeans.labels_))
            davies_bouldin_score_list.append(davies_bouldin_score(X, kmeans.labels_))

        print('max(silhouette_coefficients): ', np.argmax(silhouette_coefficients)*2+2)
        print('min(davies_bouldin_score_list): ', np.argmin(davies_bouldin_score_list)*2+2)
        plt.plot(k_range, silhouette_coefficients, label='Silhouette Coefficient')
        plt.plot(k_range, davies_bouldin_score_list, label='Davies Bouldin Score')
        plt.xticks(k_range_xtick)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Metric Score")
        plt.legend()
        filename = filenamePrefix + 'Clustering-FindK-SC-{id}.png'
        saveGraph(plt, filename, self.identifier)

    def transformKmeanClustering(self, k):
        print('Transform k-mean clustering of {title} with k = {k_value}'.format(title=self.title, k_value=k))
        X = np.copy(self.X)
        y = np.copy(self.y)
        n_clusters = k
        kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 1000,
            "random_state": RANDOM_SEED,
        }
        clusterer = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
        return clusterer.fit_transform(X)

    def analyzeKmeanClustering(self, k, filenamePrefix='', feature1=0, feature2=1):

        print('Analyzing k-mean clustering of {title} with k = {k_value}'.format(title=self.title, k_value=k))

        # customized parameters
        X = self.X
        y = self.y
        _, n_features = X.shape
        n_clusters = k

        kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 1000,
            "random_state": RANDOM_SEED,
        }

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
        cluster_labels = clusterer.fit_predict(X)

        print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        silhouette_values_list = []
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            silhouette_values_list.append(np.mean(ith_cluster_silhouette_values))

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        ax2.scatter(X[y==0, feature1], X[y==0, feature2], marker='.', label='y=0', alpha=0.8, c='blue')
        ax2.scatter(X[y==1, feature1], X[y==1, feature2], marker='.', label='y=1', alpha=0.8, c='red')

        ind = np.argpartition(silhouette_values_list, -3)[-3:]

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        centers = centers[ind]
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=0.7, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % ind[i], alpha=0.7, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the feature %d " % feature1)
        ax2.set_ylabel("Feature space for the feature %d " % feature2)

        plt.legend()
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters))

        plt.title("Silhouette analysis for KMeans clustering with k = %d" % n_clusters)
        filename = filenamePrefix + "Clustering-SilhouetteAnalysis-{id}.png"
        saveGraph(plt, filename, self.identifier)
        plt.close()

        for i in range(1, n_features):

            if i > 15:
                continue

            for j in range(1, n_features):

                if j > 15:
                    continue

                fig = plt.gcf()
                fig.set_size_inches(7, 7)
                ax = fig.add_subplot(111)

                plt.scatter(X[y==0, j], X[y==0, i], marker='.', label='y=0', alpha=0.8, c='blue')
                plt.scatter(X[y==1, j], X[y==1, i], marker='.', label='y=1', alpha=0.8, c='red')

                # Labeling the clusters
                centers = clusterer.cluster_centers_
                centers = centers[ind]
                # Draw white circles at cluster centers
                plt.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=0.7, s=200, edgecolor='k')
                for i, c in enumerate(centers):
                    plt.scatter(c[0], c[1], marker='$%d$' % ind[i], alpha=0.7, s=50, edgecolor='k')

                ax.set_title("The visualization of the clustered data.")
                ax.set_xlabel("Feature space for the feature %d " % j)
                ax.set_ylabel("Feature space for the feature %d " % i)

                plt.suptitle(("Clusters plot for k-means clustering on sample data "
                            "with n_clusters = %d" % n_clusters),
                            fontsize=14, fontweight='bold')

                plt.legend()
                filename = filenamePrefix + "Clustering-SilhouetteAnalysis-{id}-{i}-{j}.png".format(id=self.identifier, i=i, j=j)
                saveGraph(plt, 'testing/' + filename, self.identifier)
                plt.close()



    def transformEMClustering(self, k):
        print('Transform EM clustering of {title} with k = {k_value}'.format(title=self.title, k_value=k))
        X = np.copy(self.X)
        y = np.copy(self.y)
        n_clusters = k
        kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": RANDOM_SEED,
        }
        em = GaussianMixture(n_components=n_clusters, covariance_type = 'diag', random_state=RANDOM_SEED).fit(X)
        w = em.fit(X).covariances_
        newdata = X@w.T
        return newdata

    def analyzeEMClustering(self, k, filenamePrefix='', feature1=0, feature2=1):

        # customized parameters
        X = self.X
        y = self.y
        _, n_features = X.shape
        n_clusters = k

        print('Analyzing ME clustering of {title} with k = {k_value}'.format(title=self.title, k_value=k))

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED).fit(X)
        cluster_labels = clusterer.fit_predict(X)
        print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)


        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        silhouette_values_list = []
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            silhouette_values_list.append(np.mean(ith_cluster_silhouette_values))

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        ax2.scatter(X[y==0, feature1], X[y==0, feature2], marker='.', label='y=0', alpha=0.8, c='blue')
        ax2.scatter(X[y==1, feature1], X[y==1, feature2], marker='.', label='y=1', alpha=0.8, c='red')

        #ind = np.argpartition(silhouette_values_list, -3)[-3:]

        # # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # centers = centers[ind]
        # # Draw white circles at cluster centers
        # ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=0.4, s=200, edgecolor='k')
        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker='$%d$' % ind[i], alpha=0.5, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the feature %d " % feature1)
        ax2.set_ylabel("Feature space for the feature %d " % feature2)

        plt.legend()
        plt.suptitle(("Silhouette analysis for EM clustering on sample data "
                    "with n_clusters = %d" % n_clusters))

        plt.title("Silhouette analysis for EM clustering with k = %d" % n_clusters)
        filename = filenamePrefix + "EMClustering-SilhouetteAnalysis-{id}.png"
        saveGraph(plt, filename, self.identifier)
        plt.close()

        # for i in range(1, n_features):

        #     if i > 15:
        #         continue

        #     for j in range(1, n_features):

        #         if j > 15:
        #             continue

        #         fig = plt.gcf()
        #         fig.set_size_inches(7, 7)
        #         ax = fig.add_subplot(111)

        #         plt.scatter(X[y==0, j], X[y==0, i], marker='.', label='y=0', alpha=0.8, c='blue')
        #         plt.scatter(X[y==1, j], X[y==1, i], marker='.', label='y=1', alpha=0.8, c='red')

        #         ax.set_title("The visualization of the clustered data.")
        #         ax.set_xlabel("Feature space for the feature %d " % j)
        #         ax.set_ylabel("Feature space for the feature %d " % i)

        #         plt.suptitle(("Clusters plot for EM clustering on sample data "
        #                     "with n_clusters = %d" % n_clusters),
        #                     fontsize=14, fontweight='bold')

        #         plt.legend()
        #         filename = filenamePrefix + "EMClustering-SilhouetteAnalysis-{id}-{i}-{j}.png".format(id=self.identifier, i=i, j=j)
        #         saveGraph(plt, 'testing/' + filename, self.identifier)
        #         plt.close()

