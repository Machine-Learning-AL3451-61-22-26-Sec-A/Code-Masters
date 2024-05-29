import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Load the dataset
dataset = load_iris()

# Prepare the data
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Streamlit app
st.title("Iris Dataset Clustering")
st.write("This app visualizes the clustering of the Iris dataset using KMeans and Gaussian Mixture Models (GMM).")

# Show the dataset
st.write("### Iris Dataset")
st.write(X.head())

# Plotting
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# Real Plot
plt.subplot(1, 3, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real')

# KMeans Plot
plt.subplot(1, 3, 2)
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
plt.title('KMeans')

# GMM Plot
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)

plt.subplot(1, 3, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')

# Display the plot in Streamlit
st.pyplot(plt)
