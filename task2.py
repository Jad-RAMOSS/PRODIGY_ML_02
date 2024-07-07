#Jad Sherif Gad Soliman El Wahy
#Task 2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
############################Data Preprocessing
data = pd.read_csv('Mall_Customers.csv')
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


#############################Choosing the Number of Clusters

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plotting results
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



########################K-means Algorithm

kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Adding cluster labels to the original data
data['Cluster'] = clusters

# Display the first few rows of the data with cluster labels
print(data.head())

#########################Evaluating the Results

# Visualizing the clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Plotting the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='orange', label='Centroids')
plt.title('Customer Segments and Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



# Pair Plot
sns.pairplot(data, hue='Cluster', vars=['Annual Income (k$)', 'Spending Score (1-100)'])
plt.show()

