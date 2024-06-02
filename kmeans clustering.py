from sklearn.cluster import KMeans


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

