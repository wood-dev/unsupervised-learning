# <a name="_d9n8kug9zfwy"></a>Unsupervised Learning 
In this project I will be using the dataset Online Shoppers Intention and Census Income. Online Shoppers Intention (called as dataset1) captures browser-related statistics (like how much time the users spend on each page), environment and time information such as month, weekend, region to see if the visit would induce any revenue. Census Income captures the individual background data (such as age, education, and occupation) and a target boolean if his income is greater than 50k.

Both datasets contain categorical type data so one-hot encoding has been applied. The number of features have been dramatically increased to 26 and 82 respectively, I would be interested to see if they could be handled properly by clustering and dimensionality reduction. Also these two datasets have significantly different numbers of features and sample volume, that would be good for demonstrating the runtime and performance difference.
## <a name="_b1xszjgvyumd"></a>1. Run the clustering algorithms
To determine a proper value of k as the number of clustering, the experiment is using the elbow method, the silhouette coefficient, and the davies bouldin score. The elbow method is a heuristic applied on error rate to find an optimization point that the diminishing error is no longer worth the increase of cluster number. Figure 1.1 shows the elbow point = 10 where the dropping rate of error starts slowing down. Figure 1.2 shows the average silhouette coefficient and davies bouldin score which represent the performance of cluster separation. In general a high silhouette coefficient and low davies bouldin score is preferred for choosing the number of k. Figure 1.3 shows a visual sample how class can be separated through clustering.


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.001.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.002.png)|
| :-: | :-: |
|Figure 1.1 - SSE vs cluster numbers of dataset 1|Figure 1.2 - metrics vs cluster numbers of dataset 1|

In this experiment, I found the elbow point is not clear compared with the latter so it’s not considered, and the best k of dataset1 and dataset2 are 20 and 52 respectively. The numbers are higher than expected I shall see if there is any room for improvement in later experiments.

Figure 2.1 plots the silhouette score of different clusters and the visualization of raw data with k-mean clustering. With the optimal value of k, the average silhouette coefficient is 0.341616, the value is not high and some negative value in the clusters may imply the clustering is not ideal. The analysis on the right shows how the data are located between cluster 16 and cluster 22, and the centroids of clusters with top 3 average silhouette scores. The separation between cluster 8 and 19 is not clear, but cluster 10 can be used to classify the points where y=0. Figure 2.2 shows another pair of features consistent with figure 2.1, where the points with y=1 closer to cluster 8 and cluster 19, and points with y=0 are closer to cluster 10. This is also consistent with the result of Supervised Learning, where the feature 17 (which is Month\_Nov) has put a great contribution on determining the outcome. 


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.003.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.004.png)|
| :-: | :- |
|Figure 2.1 - silhouette plot and analysis for dataset 1|Figure 2.2 - data visualization on feature 5 vs feature 17|


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.005.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.006.png)|
| :-: | :-: |
|Figure 3.1 - data visualization on feature 5 vs feature 16|Figure 3.2 - data visualization on  feature 5 vs feature 17|

Another experiment using Expectation Maximization (EM) clustering shows visually consistent graphs on data as k-mean in figure 3.1. However the clustering is not well with a lot of negative silhouette scores in some clusters. Also there are differences in the ground truth based NMI score and average silhouette score. Apparently the better average silhouette score does not necessarily result in a better correlation between target prediction and truth, also some clusters have negative silhouette that indicates the clustering is not ideal.


|||**K-means clustering**|**EM clustering**|
| :-: | :-: | :-: | :-: |
|**dataset**|**Cluster no.**|**NMI score**|**Average silhouette**|**NMI score**|**Average silhouette**|
|**1**|20|0\.017054|0\.341616|0\.037934|0\.192330|
|**2**|52|0\.068081|0\.264219|0\.066880|0\.263833|

Table 1.1 - clustering statistics

Next the same procedure has been applied on dataset 2. Due to the large number of k the silhouette plot is not clear in figure 4.1. It is found the centroids of clusters are too close, a bad clustering is formed. Another interesting findings is when most features are categorical, scattering cannot show the data distribution well as in figure 4.2. Also noted in figure 4.2, the plot of points shown on feature 5 vs feature 7 is consistent with the decision tree in Supervised Learning, whereas education-num and marital-status are dominant factors.

The experiments above have been repeated for a better clustering result hopefully. However, there could be no better combination of clustering graphs and metrics found. So in the following experiment I would continue the journey with the clustering method used.

|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.007.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.008.png)|
| :- | :- |
|Figure 4.1 - silhouette plot and analysis for dataset 1|Figure 4.2 - data visualization on  feature 5 vs feature 7|
## <a name="_v9dvuo2d8hu"></a>2. Dimensionality reduction algorithms
In following experiments, 2-steps processes are done on experiments across PCA, ICA, RP, and Truncated SVD. I would first try finding the optimal component numbers, followed by analysis with its runtime and scattering plot. 

For PCA, the best component number is identified by looking at its variance and eigenvalue of each component, in figure 5.1 and 5.2. I have done the experiments repeatedly and found the data points are close and mostly overlapping so I would need restoring variance and keeping relevant information intact. A variance close to 0.99 has been taken. Also note from figure 5.2 the eigenvalues of the initial components are more dominant, then dropping at a slightly decreasing rate because they are shared across subsequent components.


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.009.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.010.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.011.png)|
| :-: | :-: | :-: |
|Figure 5.1 - PCA Cumulative Variance|Figure 5.2 - PCA Eigenvalues|Figure 5.3 Data visualization|



|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.012.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.013.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.014.png)|
| :-: | :-: | :-: |
|Figure 6.1 - ICA Kurtosis|Figure 6.2 - Reconstruction Error|Figure 6.3 Data visualization|

Similar procedures have been done on ICA. Instead of looking into variance, kurtosis and error are used to find out the optimal value of component number. As we want to have the component best representing data independency with minimal error, optimal points would be found at high kurtosis and low error. Note the average kurtosis increases significantly only at a higher number of components. The optimal number of components have been identified as 24 and 78 for dataset 1 and dataset 2 respectively. They are quite close to the original number of features, implying the data feature dimensionality cannot be reduced well to independent Gaussian components. Resulted graph on dataset 2, in figure 6.3 shows the data is overlapping. 

For RP and TSVD, experiments on reconstruction error are done to deduce the optimal number of components. Multiple runs with average value is taken on RP to show its error versus number of components, showing they are inversely correlated in a linear manner as in figure 7.1. Truncated SVD on the other hand shows exponential decline such that lower error can be obtained with fewer numbers of components. Its performance is similar to PCA, while they both make use of factorization, TSVD is done on data matrix instead of covariance matrix. So their data distributions are similar as in figure 5.3 and 8.3.


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.015.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.016.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.017.png)|
| :-: | :-: | :-: |
|Figure 7.1 - RP Construction Error|Figure 7.2 - SVD Reconstruction Error|Figure 7.3 SVD Reconstruction Error|


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.018.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.019.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.020.png)|
| :-: | :-: | :-: |
|Figure 8.1 dataset 1 RP data|Figure 8.2 dataset 2 RP data|Figure 8.3 dataset 1 TSVD data|

The optimal result of dimensionality reduction algorithms is summarized below. It is reasonable the runtime is proportional to the component number. ICA takes a long time to identify independence and RP is the quickest due to its random nature. Interestingly TSVD is much quicker than PCA on dataset 1, perhaps the calculation of singular value is quicker than covariance.

||**Dataset 1 - online shopper intention**|**Dataset 2 - census income**|
| :-: | :-: | :-: |
||**component number**|**runtime**|**Component number**|**runtime**|
|PCA|19|0\.214900|50|0\.407561|
|ICA|24|1\.740620|78|4\.848825|
|RP|22|0\.012921|78|0\.037992|
|Truncated SVD|16|0\.058967|50|0\.433730|
## <a name="_cwutq7fpp8b9"></a>3. Reproduce clustering experiments after dimensionality reductions
After dimensionality reductions the number of features are changed so a similar step to find the optimal number of clusters, by looking at the silhouette coefficient and davies bouldin score. Note the component number for each DR is taken from the result of part 2. 


|**DR** |**Cluster No**|**K-means clustering**|**EM clustering**|
| :-: | :-: | :-: | :-: |
|||**NMI score**|**Avg silhouette**|**NMI score**|**Avg silhouette**|
|**PCA**|16|0\.017369|0\.343401|0\.018668|0\.265687|
|**ICA**|20|0\.027106|0\.210698|0\.038740|0\.139081|
|**RP**|16|0\.015744|0\.362924|0\.027810|0\.295751|
|**TSVD**|10|0\.020474|0\.342419|0\.013066|0\.355126|

Table 3.1 - Clustering after dimensionality reductions on dataset 1


|**DR** |**Cluster No**|**K-means clustering**|**EM clustering**|
| :-: | :-: | :-: | :-: |
|||**NMI score**|**Avg silhouette**|**NMI score**|**Avg silhouette**|
|**PCA**|48|0\.068583|0\.262047|0\.067106|0\.256511|
|**ICA**|4|0\.005299|0\.510985|0\.000207|0\.229528|
|**RP**|76|0\.062942|0\.298430|0\.067113|0\.298658|
|**TSVD**|44|0\.067813|0\.258202|0\.070033|0\.243274|

Table 3.2 - Clustering after dimensionality reductions on dataset 2

By comparing the futures from table 3.1, 3.2 and the one of table 1.1 that was done with no DR. There are a few interesting findings:

- Except for the RP of dataset 2, the clustering usually requires fewer numbers of clusters after DR, this is sensible as there are fewer number of features.
- The RP of dataset 2 shows the clustering is bad and the data points are not separable properly in figure 9.1. The random principle cannot transform the data properly into new clusters.
- For dataset 2, the NMI scores are generally consistent between k-means and EM clustering, perhaps the data is sparse so the nearest neighbour is consistent with the density probability.
- Generally the clustering after some of the DRs are able to maintain a better NMI score and average silhouette, that means the performance can be maintained even with some features removed. 
- For dataset 2, the optimal cluster number is found as 4 and that actually resulted in a quite good average silhouette score. However from figure 9.2, the centroids are actually too close, and most of the data points are in cluster 3, that resulted in a bad clustering result. That could also be the reason for the low NMI score. 
- In contrast, the result of PCA on dataset 1 has shown a better result, in figure 9.3. While its NMI score and average silhouette are maintained, it shows reasonable clustering and the data visualization has shown data can be better classified compared with figure 2.1.
- ICA shows a good NMI score after clustering but is the opposite case in dataset 2, that means transforming dataset 1 to independence component can present a better result.


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.021.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.022.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.023.png)|
| :-: | :-: | :-: |
|Figure 9.1 |Figure 9.2 |Figure 9.3 |


Some samples on data visualization after DR are shown below, here are some of my observations.

- In figure 10.1, the clustering after PCA shows data can be separated into the groups with centroid slightly above and below zero / or positive and negative, which is more separable than that before PCA in figure 2.1. Also the multiple sections in the graph represents the need of a cluster number of 16. Besides, we can also see how data is transformed to different axes.
- From figure 10.2, it is observed on the same transformation and dataset, some features are more classificable.
- In figure 10.3, the high number of components resulted in a sparse data distribution that cannot be easily classified through clustering.

|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.024.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.025.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.026.png)|
| :-: | :-: | :-: |
|Figure 10.1 - PCA on dataset1|Figure 10.2 - PCA on dataset1|Figure 10.3 - PCA on dataset2|

- In figure 10.4, it shows ICA can simplify some of the resulting data points.
- So far it is often seen the target y=0 can be separated but y=1 points are usually overlapping the opposite class, I would expect fewer negative target can be correctly identified due to the ambiguity. The situation is obvious in figure 10.5.
- When clusters are separated with proper distance, the data points are easily classified as in figure 10.6.

|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.027.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.028.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.029.png)|
| :-: | :-: | :-: |
|Figure 10.4 - ICA on dataset1|Figure 10.5 - ICA on dataset1|Figure 10.6 - ICA on dataset2|


- In figure 10.7, RP tends to create more spare data distribution compared with other transformations. I found it harder to perform clustering.
- In figure 10.8, RP on categorical data of dataset 2 resulted in scaling the features. Some features may be projected evenly as in figure 10.9.
- In figure 10.10 TSVD has similar scattering as PCA as seen in figure 10.1. 
- From 10.11, we can see that data which is properly clustered may not necessarily produce better classification as different classes of data can be overlapping.
- Figure 10.12 shows that sometimes data could simply cannot be separated even with the optimal numbers of transformation components and clustering.

|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.030.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.031.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.032.png)|
| :-: | :-: | :-: |
|Figure 10.7 - RP on dataset1|Figure 10.8 - RP on dataset2|Figure 10.9 - RP on dataset2|


|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.033.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.034.png)|![](graph/5cd988c6-9a3e-4cb2-a923-425490a5fbf7.035.png)|
| :-: | :-: | :-: |
|Figure 10.10 - TSVD on dataset1|Figure 10.11 - TSVD on dataset1|Figure 10.12 - TSVD on dataset2|

## <a name="_mzit7jg34ymu"></a>4. Apply the DR then rerun Neural Network Learner on dataset 1
First here is the result of neural network learner (NN) on dataset 1 without DR, for further comparison with later experiments. The layer setting used is **(20, 20)**, and the solver is **adam**.


|**Confusion Matrix**|**Classification Report**|**NMI Score**|**Training Time**|**Query Time**|
| :-: | :-: | :-: | :-: | :-: |
|<p>[[973  44]</p><p>[ 95 121]]</p>|<p>precision    recall  f1-score   support</p><p>0       0.91      0.96      0.93      1017</p><p>1       0.72      0.54      0.62       216</p><p>accuracy                           0.88      1233   </p><p>weighted avg       0.87      0.88      0.88      1233</p>|0\.294919|22\.879481|0\.005761|

The transformed dataset has different dimensions and data so a re-analysis is required on each DR transformation for obtaining the optimal parameter settings. The identified optimal parameter and its neutral networking running result is shown below, with **adam** found as a better solver in all cases.


|**DR**|**Layers**|**Confusion Matrix**|**Classification Report**|**NMI** |**Training** |**Query** |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|PCA|{5}|<p>[[1006   11]</p><p>[ 194   22]]</p><p></p>|<p>precision    recall  f1-score   support</p><p>0       0.84      0.99      0.91      1017</p><p>1       0.67      0.10      0.18       216</p><p>accuracy                           0.83      1233   </p><p>weighted avg       0.81      0.83      0.78      1233</p>|0\.055818|5\.524063 |0\.000999 |
|ICA|{15,15}|<p>[[975  42]</p><p>[102 114]]</p><p></p><p></p>|<p>precision    recall  f1-score   support</p><p>0       0.91      0.96      0.93      1017</p><p>1       0.73      0.53      0.61       216</p><p>accuracy                           0.88      1233</p><p>weighted avg       0.87      0.88      0.88      1233</p>|0\.276521|19\.689275 |0\.002774 |
|RP|{20,20}|<p>[[984  33]</p><p>[123  93]]</p><p></p><p></p>|<p>precision    recall  f1-score   support</p><p>0       0.89      0.97      0.93      1017</p><p>1       0.74      0.43      0.54       216</p><p>accuracy                           0.87      1233   </p><p>weighted avg       0.86      0.87      0.86      1233</p>|0\.231928|21\.824537 |0\.002507 |
|TSVD|{10}|<p>[[1012    5]</p><p>[ 210    6]]</p><p></p>|<p>precision    recall  f1-score   support</p><p>0       0.83      1.00      0.90      1017</p><p>1       0.55      0.03      0.05       216</p><p>accuracy                           0.83      1233</p><p>weighted avg       0.78      0.83      0.75      1233</p>|0\.012229|5\.744768 |0\.000998 |

Overall we can see layers with fewer nodes and layers are needed in general, as the dataset has been simplified through DR. All confusion matrices show a lower number in negative conclusion, I would suggest it is due to the imbalance nature of the dataset - with a significantly lower ratio of negative samples given, the dimensionality reduced dataset would polarize the result. The finding is also consistent with the graphs in figure 10, where a lot of negative samples are found overlapping with positive ones that can be hardly classified. From the classification report, the overall weighted accuracy and precision are lower but the result of ICA is comparable with the original result, which can be reflected from the NMI score too. Moreover we can see improvement in terms of training and query time after DR. RP takes the longest training time as the data is not transformed well as other algorithms. PCA has shaped the data well hence reducing the training and query time but accuracy has been sacrificed due to some reasons.
## <a name="_aeu7q1o3ry9l"></a>5. Apply the DR then clustering, rerun Neural Network Learner on dataset 1
In this section I am taking the result from part 3, where dataset 1 has been processed through different algorithms of dimensionality reduction followed by clusterings. The results are later passed to the neural network learner, with similar parameters finding process. 

As a recap of parameters used and optimal parameters found are summarized here.


<table><tr><th valign="top"></th><th valign="top"><b>n_components</b></th><th valign="top"><b>n_clusters</b></th><th valign="top"><b>Clustering</b></th><th valign="top"><b>best layers found</b></th><th valign="top"><b>solver</b></th></tr>
<tr><td rowspan="2" valign="top"><b>PCA</b></td><td rowspan="2" valign="top">19</td><td rowspan="2" valign="top">16</td><td valign="top"><b>k-means</b></td><td valign="top">{5}</td><td valign="top">adam</td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top">{20, 20}</td><td valign="top">adam</td></tr>
<tr><td rowspan="2" valign="top"><b>ICA</b></td><td rowspan="2" valign="top">24</td><td rowspan="2" valign="top">20</td><td valign="top"><b>k-means</b></td><td valign="top">{10, 10, 10, 10}</td><td valign="top">adam</td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top">{5}</td><td valign="top">adam</td></tr>
<tr><td rowspan="2" valign="top"><b>RP</b></td><td rowspan="2" valign="top">22</td><td rowspan="2" valign="top">16</td><td valign="top"><b>k-means</b></td><td valign="top">{25, 25, 25}</td><td valign="top">adam</td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top">{25, 25}</td><td valign="top">adam</td></tr>
<tr><td rowspan="2" valign="top"><b>TSVD</b></td><td rowspan="2" valign="top">16</td><td rowspan="2" valign="top">10</td><td valign="top"><b>k-means</b></td><td valign="top">{5}</td><td valign="top">adam</td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top">{5}</td><td valign="top">adam</td></tr>
</table>

As a quick glance, the neural network has layers completely different from part 4, the experiments with no clustering. It is because the neural network now processes the clustered group of data as features now.

Below is the table showing the details of neural network results.


<table><tr><th valign="top"><b>DR</b></th><th valign="top"><b>Cluster</b></th><th valign="top"><b>Confusion Matrix</b></th><th valign="top"><b>Classification Report</b></th><th valign="top"><b>NMI</b> </th><th valign="top"><b>Training</b> </th><th valign="top"><b>Query</b> </th></tr>
<tr><td rowspan="2" valign="top"><b>PCA</b></td><td valign="top"><b>k-means</b></td><td valign="top"><p>[[1017    0]</p><p>[ 216    0]]</p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.82      1.00      0.90      1017</p><p>1       0.00      0.00      0.00       216</p><p>accuracy                           0.82      1233</p><p>weighted avg       0.68      0.82      0.75      1233</p></td><td valign="top">0</td><td valign="top">8\.528515</td><td valign="top">0\.000999</td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top"><p>[[1013    4]</p><p>[ 211    5]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.83      1.00      0.90      1017</p><p>1       0.56      0.02      0.04       216</p><p>accuracy                           0.83      1233</p><p>weighted avg       0.78      0.83      0.75      1233</p></td><td valign="top">0\.010652</td><td valign="top">13\.157904</td><td valign="top">0\.001998</td></tr>
<tr><td rowspan="2" valign="top"><b>ICA</b></td><td valign="top"><b>k-means</b></td><td valign="top"><p>[[988  29]</p><p>[114 102]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.90      0.97      0.93      1017</p><p>1       0.78      0.47      0.59       216</p><p>accuracy                           0.88      1233</p><p>weighted avg       0.88      0.88      0.87      1233</p></td><td valign="top">0\.275569</td><td valign="top">12\.377059 </td><td valign="top">0\.001998 </td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top"><p>[[1017    0]</p><p>[ 216    0]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.82      1.00      0.90      1017</p><p>1       0.00      0.00      0.00       216</p><p>accuracy                           0.82      1233</p><p>weighted avg       0.68      0.82      0.75      1233</p></td><td valign="top">0</td><td valign="top">0\.986506 </td><td valign="top">0</td></tr>
<tr><td rowspan="2" valign="top"><b>RP</b></td><td valign="top"><b>k-means</b></td><td valign="top"><p>[[1005   12]</p><p>[ 191   25]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.84      0.99      0.91      1017</p><p>1       0.68      0.12      0.20       216</p><p>accuracy                           0.84      1233</p><p>weighted avg       0.81      0.84      0.78      1233</p></td><td valign="top">0\.063829</td><td valign="top">18\.707347 </td><td valign="top">0\.002999 </td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top"><p>[[1013    4]</p><p>[ 204   12]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.83      1.00      0.91      1017</p><p>1       0.75      0.06      0.10       216</p><p>accuracy                           0.83      1233</p><p>weighted avg       0.82      0.83      0.77      1233</p></td><td valign="top">0\.039297</td><td valign="top">36\.092196 </td><td valign="top">0\.003998 </td></tr>
<tr><td rowspan="2" valign="top"><b>TSVD</b></td><td valign="top"><b>k-means</b></td><td valign="top"><p>[[1017    0]</p><p>[ 216    0]]</p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.82      1.00      0.90      1017</p><p>1       0.00      0.00      0.00       216</p><p>accuracy                           0.82      1233</p><p>weighted avg       0.68      0.82      0.75      1233</p></td><td valign="top">0</td><td valign="top">7\.326169 </td><td valign="top">0</td></tr>
<tr><td valign="top"><b>EM</b></td><td valign="top"><p>[[1017    0]</p><p>[ 216    0]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.82      1.00      0.90      1017</p><p>1       0.00      0.00      0.00       216</p><p>accuracy                           0.82      1233</p><p>weighted avg       0.68      0.82      0.75      1233</p></td><td valign="top">0</td><td valign="top">4\.540879 </td><td valign="top">0</td></tr>
<tr><td valign="top"><b>(original)</b></td><td valign="top"><b>-</b></td><td valign="top"><p>[[973  44]</p><p>[ 95 121]]</p><p></p><p></p></td><td valign="top"><p>precision    recall  f1-score   support</p><p>0       0.91      0.96      0.93      1017</p><p>1       0.72      0.54      0.62       216</p><p>accuracy                           0.88      1233   </p><p>weighted avg       0.87      0.88      0.88      1233</p></td><td valign="top">0\.294919</td><td valign="top">22\.879481</td><td valign="top">0\.005761</td></tr>
</table>

- It is observed that some models can only conclude positive results. Taking PCA as an example, after DR, some noises and apparently irrelevant or useless data are removed. Clustering has further put them into groups but as seen in part 3 the negatives are just overlapping with the positives. As a result the positives as the majority has dominated the clusters and give the result as positive. The zero NMI result has further shown the result is not correlated to the ground truth anymore.
- EM-Clustering of PCA has retained some of the negative outcome, as it is taking all samples for density probability calculation, instead of just k nearest samples.
- ICA is showing the opposite result on k-means and EM, perhaps its transformation has retained the independence of features that favours the k-means clustering
- With all information retained, it is expected the result from EM clustering would have longer training time. It is also observed that more layers are used for neural network training.
- While RP takes longer time on training, its transformation has actually retained partial information for both positive and negative outcomes. For this setting we can see that despite RP itself is fast, the transformed dataset would need more time on training.
- The k-mean clustering of ICA has the best performance among all other settings in terms of overall accuracy and precision, it also has the highest NMI score. It has a comparable result with the original neural network learner (without dimensionality reduction and clustering), with improved training time and query time. Also we can see from part 3 there is visibility on data in terms of different features as the advantages of this experiment.
- Apparently the feature values have Gaussian distribution so ICA can have reasonable performance.
- However the failures of some clustering in this part and part 3 may be due to imbalance in the dataset, data sparsity, or some significant outliers being removed in the reduction phase.

## <a name="_x932szno6yil"></a>Running under Anaconda
1. The environment file for anaconda is environment.yml
1. In UnsupervisedLearning.py, there are 5 sets of key boolean variables default to False. 
From its naming, each variable partN is to enable execution of each part. 

		part1 = False
		findingOptimal = False
	
		part2 = False
		findingOptimal = False
	
		part3 = False
		findingOptimal = False
	
		part4 = False
		findingOptimal = False
	
		part5 = False
	findingOptimal = False
	
What has been already done on each part is:
- First, set findingOptimal to True, it will generate console output and / or graphs for identifying optimal parameter. 
- Then the identified optimal parameters are coded in the subsequent coding of UnsupervisedLearning.py. Next set findingOptimal to False, then rerun that part to perform detail analysis for data plot / runtime / classification report.