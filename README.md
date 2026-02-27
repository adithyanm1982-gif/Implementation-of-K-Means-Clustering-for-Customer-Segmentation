# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset containing customer details.
2. Select relevant features (Annual Income and Spending Score) for clustering.
3. Initialize the K-Means model with a predefined number of clusters (k = 5) and fit the model to the selected data.
4. Assign cluster labels to each data point and append the cluster information to the dataset.
5. Visualize the clusters along with their centroids using a scatter plot to analyze customer segmentation. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ADITHYA NM
RegisterNumber:  212225040011
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(data.head())
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)


data['Cluster'] = y_kmeans

print("\nClustered Data:")
print(data.head())


plt.figure()
plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'], 
            X[y_kmeans == 0]['Spending Score (1-100)'], label='Cluster 0')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'], 
            X[y_kmeans == 1]['Spending Score (1-100)'], label='Cluster 1')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'], 
            X[y_kmeans == 2]['Spending Score (1-100)'], label='Cluster 2')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'], 
            X[y_kmeans == 3]['Spending Score (1-100)'], label='Cluster 3')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'], 
            X[y_kmeans == 4]['Spending Score (1-100)'], label='Cluster 4')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=200, label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```
## Output:

<img width="692" height="139" alt="ML-EX-10-A" src="https://github.com/user-attachments/assets/1d82e517-82ce-4b4c-909a-556c13b8173b" />

## Clustered Data
<img width="765" height="346" alt="ML-EX-10-B" src="https://github.com/user-attachments/assets/6d674707-e05f-42c4-ab8a-f64bc4781522" />

## Customer Segmentation using K-Means
<img width="842" height="583" alt="ML-EX-10-C" src="https://github.com/user-attachments/assets/34d65ec9-a06f-40f4-84fe-d715a3956a96" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
