# K-Means Clustering ALgorithm tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 06OCT20

# Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

# Import dataset (there is no dependent variable (y-variable) a priori)
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Use the Elbow Method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show();

# Train the K-Means model on the dataset
    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    

# Visualize the clusters (we need a 2-Dimensional plot!)