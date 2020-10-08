# K-Means Clustering ALgorithm tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 06OCT20

# Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset (there is no dependent variable (y-variable) a priori)
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Use the Elbow Method to find the optimal number of clusters


# Train the K-Means model on the dataset


# Visualize the clusters (we need a 2-Dimensional plot!)