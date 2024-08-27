import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


X = pd.read_csv('./.../feature.csv')['feature_name'].dropna().values.reshape(-1, 1)

k_values = range(min_num, max_num)
inertia, centroids, rate = [], [], []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    inertia_values.append(kmeans.inertia_)
    centroids.append(kmeans.cluster_centers_)
for i in range(0, len(inertia)-1):
    rate[i+1] = (inertia[i]-inertia[i+1])/inertia[i]
    print(rate)

plt.gcf().subplots_adjust(bottom=0.15)
plt.plot(k_values, inertia, color='black')
plt.plot(k_values, inertia, marker='s', color='none', markeredgecolor='red', markeredgewidth=2)
plt.xlabel('Number of Clusters (${L}$)', font={'family': 'TImes New Roman', 'size': 16})
plt.ylabel('Sum of Squared Errors (SSE)', font={'family': 'TImes New Roman', 'size': 16})
plt.xticks(fontproperties='TImes New Roman', size=16)
plt.yticks(fontproperties='TImes New Roman', size=16)
plt.show()
